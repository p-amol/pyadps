"""
Reusable helpers for exporting the raw (unprocessed) ADCP dataset returned
by ``pyadps.read()`` - i.e. before any ``ProcessedDataset`` wrapping/QC.

Shared by the Download Raw File page (interactive, full component picker)
and ``autoprocess()`` (config.ini-driven reprocessing), so the two stay in
sync instead of maintaining separate copies of the same subsetting logic.
"""

import xarray as xr

#: Internal/metadata attrs that only make sense while the dataset is still
#: part of the read/merge pipeline - not meaningful once a subset is
#: written out as a standalone file.
INTERNAL_ATTRS = (
    "pyadps_component",
    "components",
    "fixed_leader_variables",
    "variable_leader_variables",
)


def subset_raw_dataset(
    ds: xr.Dataset,
    include_fixed_leader: bool = False,
    include_variable_leader: bool = False,
    include_velocity: bool = False,
    include_echo: bool = False,
    include_correlation: bool = False,
    include_percent_good: bool = False,
) -> xr.Dataset:
    """
    Build a subset of the raw dataset containing only the selected
    components - the same choice as the Download Raw File page's
    checkboxes (checking all six is equivalent to its "Entire Data Set"
    option).

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from ``pyadps.read()``.
    include_fixed_leader : bool, default False
        Include Fixed Leader variables (from ``ds.attrs
        ['fixed_leader_variables']``).
    include_variable_leader : bool, default False
        Include Variable Leader variables (from ``ds.attrs
        ['variable_leader_variables']``).
    include_velocity : bool, default False
        Include the 'velocity' variable.
    include_echo : bool, default False
        Include the 'echo_intensity' variable.
    include_correlation : bool, default False
        Include the 'correlation' variable.
    include_percent_good : bool, default False
        Include the 'percent_good' variable.

    Returns
    -------
    xr.Dataset
        Subset dataset with matching coordinates and global attributes.

    Raises
    ------
    ValueError
        If no requested component matches an actual variable in ``ds``.
    """
    fl_fields = ds.attrs.get("fixed_leader_variables", [])
    vl_fields = ds.attrs.get("variable_leader_variables", [])

    variables_to_include = []

    if include_fixed_leader:
        variables_to_include += [v for v in fl_fields if v in ds.data_vars]
    if include_variable_leader:
        variables_to_include += [v for v in vl_fields if v in ds.data_vars]
    if include_velocity and "velocity" in ds.data_vars:
        variables_to_include.append("velocity")
    if include_echo and "echo_intensity" in ds.data_vars:
        variables_to_include.append("echo_intensity")
    if include_correlation and "correlation" in ds.data_vars:
        variables_to_include.append("correlation")
    if include_percent_good and "percent_good" in ds.data_vars:
        variables_to_include.append("percent_good")

    # Remove duplicates while preserving order
    variables_to_include = list(dict.fromkeys(variables_to_include))

    if not variables_to_include:
        raise ValueError("No variables selected for raw export.")

    subset_ds = ds[variables_to_include].copy()

    # Copy over coordinates not already pulled in by the variable subset
    for coord in ds.coords:
        if coord not in subset_ds.coords:
            subset_ds = subset_ds.assign_coords({coord: ds.coords[coord]})

    subset_ds.attrs = ds.attrs.copy()

    return subset_ds


def drop_internal_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Drop read/merge-pipeline-only attrs (see ``INTERNAL_ATTRS``) that
    aren't meaningful once the dataset is written out as a standalone
    file. Returns a new dataset; ``ds`` itself is not modified.
    """
    ds = ds.copy()
    for attr in INTERNAL_ATTRS:
        ds.attrs.pop(attr, None)
    return ds
