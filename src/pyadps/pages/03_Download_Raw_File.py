"""
03_Download_Raw_File.py - Download Raw ADCP Data

Refactored for pyadps v1.0.0 compatibility.
Provides checkbox-based selection for downloading data components as NetCDF.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

# Load default attribute definitions from shared config
_ATTR_JSON = os.path.join(os.path.dirname(__file__), "..", "default_attributes.json")
with open(_ATTR_JSON) as _f:
    DEFAULT_ATTRIBUTES = json.load(_f)

# =============================================================================
# SESSION STATE CHECK
# =============================================================================

if "ds" not in st.session_state or st.session_state.ds is None:
    st.error("⚠️ No data loaded! Please read a file on the **Read File** page first.")
    st.stop()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "fname" not in st.session_state:
    st.session_state.fname = "No file selected"

if "attributes" not in st.session_state:
    st.session_state.attributes = {}

# raw_custom_attributes: user-added ad-hoc key-value pairs on the raw download page
if "raw_custom_attributes" not in st.session_state:
    st.session_state.raw_custom_attributes = {}

if "raw_custom_attr_count" not in st.session_state:
    st.session_state.raw_custom_attr_count = 0

if "add_attributes_DRW" not in st.session_state:
    st.session_state.add_attributes_DRW = "No"

if "filename" not in st.session_state or not st.session_state.filename:
    raw_basename = os.path.basename(st.session_state.fname)
    st.session_state.filename = os.path.splitext(raw_basename)[0]

if "file_prefix" not in st.session_state:
    st.session_state.file_prefix = st.session_state.filename

if "prefix_saved" not in st.session_state:
    st.session_state.prefix_saved = False

if "axis_option_DRW" not in st.session_state:
    st.session_state.axis_option_DRW = "time"

# =============================================================================
# LOAD DATA FROM SESSION STATE
# =============================================================================

ds = st.session_state.ds
proc = st.session_state.get("processor")

# Get field lists from dataset attributes
fl_fields = ds.attrs.get("fixed_leader_variables", [])
vl_fields = ds.attrs.get("variable_leader_variables", [])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_prefixed_filename(base_name: str) -> str:
    """Generates the file name with the optional prefix."""
    if st.session_state.file_prefix:
        return f"{st.session_state.file_prefix}_{base_name}"
    return base_name


def create_subset_dataset(
    include_fixed_leader: bool = False,
    include_variable_leader: bool = False,
    include_velocity: bool = False,
    include_echo: bool = False,
    include_correlation: bool = False,
    include_percent_good: bool = False,
) -> xr.Dataset | None:
    """
    Create a subset of the dataset based on selected components.

    Parameters
    ----------
    include_fixed_leader : bool
        Include Fixed Leader variables
    include_variable_leader : bool
        Include Variable Leader variables
    include_velocity : bool
        Include velocity data
    include_echo : bool
        Include echo intensity data
    include_correlation : bool
        Include correlation data
    include_percent_good : bool
        Include percent good data

    Returns
    -------
    xr.Dataset | None
        Subset dataset containing only selected variables, or None if nothing selected
    """
    variables_to_include = []

    # Fixed Leader variables
    if include_fixed_leader:
        for var in fl_fields:
            if var in ds.data_vars:
                variables_to_include.append(var)

    # Variable Leader variables
    if include_variable_leader:
        for var in vl_fields:
            if var in ds.data_vars:
                variables_to_include.append(var)

    # Primary data arrays
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
        st.warning("No variables selected for export.")
        return None

    # Create subset dataset
    subset_ds = ds[variables_to_include].copy()

    # Copy over coordinates
    for coord in ds.coords:
        if coord not in subset_ds.coords:
            subset_ds = subset_ds.assign_coords({coord: ds.coords[coord]})

    # Copy over global attributes
    subset_ds.attrs = ds.attrs.copy()

    return subset_ds


def add_user_attributes(dataset: xr.Dataset) -> xr.Dataset:
    """
    Add user-specified attributes to the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add attributes to

    Returns
    -------
    xr.Dataset
        Dataset with added attributes
    """
    # Add standard attributes
    for key, value in st.session_state.attributes.items():
        if value:  # Only add non-empty values
            # Convert date objects to string
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            dataset.attrs[key] = value

    # Add custom attributes
    for key, value in st.session_state.raw_custom_attributes.items():
        if key and value:  # Only add if both key and value are non-empty
            dataset.attrs[key] = value

    return dataset


def write_netcdf(
    dataset: xr.Dataset,
    axis_option: str = "time",
    add_attributes: bool = False,
) -> str:
    """
    Write dataset to NetCDF file.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to write
    axis_option : str
        X-axis option: "time" or "ensemble"
    add_attributes : bool
        Whether to add user attributes

    Returns
    -------
    str
        Path to the created NetCDF file
    """
    # Create a copy to avoid modifying the original
    ds_out = dataset.copy()

    # Add user attributes if requested
    if add_attributes:
        ds_out = add_user_attributes(ds_out)

    # Handle axis option - rename time dimension if using ensemble
    if axis_option == "ensemble":
        # If time coordinate exists, we might want to keep it as a variable
        # but use ensemble as the primary dimension
        if "time" in ds_out.dims:
            # Store time as a variable if it's currently a coordinate
            if "time" in ds_out.coords:
                time_values = ds_out.coords["time"].values
                ds_out = ds_out.rename({"time": "ensemble"})
                ds_out["time_original"] = ("ensemble", time_values)
                ds_out["time_original"].attrs["long_name"] = "Original time values"

    # Drop internal/metadata attributes that shouldn't be in the output file
    attrs_to_drop = [
        "pyadps_component",
        "components",
        "fixed_leader_variables",
        "variable_leader_variables",
    ]
    for attr in attrs_to_drop:
        if attr in ds_out.attrs:
            del ds_out.attrs[attr]

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    filename = get_prefixed_filename("RAW_DATA.nc")
    filepath = os.path.join(temp_dir, filename)

    # Write to NetCDF
    ds_out.to_netcdf(filepath)

    return filepath


def download_csv_with_ensemble(data: dict, filename: str):
    """Download data as CSV with ensemble numbers."""
    ensembles = np.arange(1, len(next(iter(data.values()))) + 1)
    df = pd.DataFrame(data)
    df.insert(0, "RDI_Ensemble", ensembles)
    csv = df.to_csv(index=False).encode("utf-8")
    return st.download_button(
        label=f"Download {filename} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
    )


def download_csv(data: dict, filename: str):
    """Download data as CSV."""
    df = pd.DataFrame.from_dict(data, orient="index").T
    csv = df.to_csv(index=False).encode("utf-8")
    return st.download_button(
        label=f"Download {filename} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
    )


def download_csv_2d(data: np.ndarray, filename: str):
    """Download 2D data as CSV with ensemble and cell indices."""
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient="index").T
    else:
        df = pd.DataFrame(data)

    ensembles = np.arange(1, df.shape[0] + 1)
    cells = np.arange(1, df.shape[1] + 1)

    df.insert(0, "Ensemble", ensembles)
    df = df.T
    df.insert(0, "Cell", [""] + list(cells))

    csv = df.to_csv(index=False, header=False).encode("utf-8")
    return st.download_button(
        label=f"Download {filename} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
    )


# =============================================================================
# UI - NETCDF FILE DOWNLOAD
# =============================================================================

_, _col_fname = st.columns([5, 1])
with _col_fname:
    st.caption(f"📂 {st.session_state.get('fname', 'No file selected')}")
st.header("NetCDF File", divider="blue")

# Option to add attributes
st.session_state.add_attributes_DRW = st.radio(
    "Do you want to add attributes to the NetCDF file?",
    ["No", "Yes"],
    horizontal=True,
)

if st.session_state.add_attributes_DRW == "Yes":
    st.write("### Please fill in the attributes:")

    # Two-column layout driven by default_attributes.json
    col1, col2 = st.columns(2)
    col_map = {1: col1, 2: col2}

    for field in DEFAULT_ATTRIBUTES:
        with col_map[field["column"]]:
            existing = st.session_state.attributes.get(field["key"], "")
            if field["widget"] == "date_input":
                st.session_state.attributes[field["key"]] = st.date_input(
                    field["label"],
                    value=existing if existing else None,
                    key=f"drw_attr_{field['key']}",
                )
            elif field["widget"] == "text_area":
                st.session_state.attributes[field["key"]] = st.text_area(
                    field["label"],
                    value=existing,
                    key=f"drw_attr_{field['key']}",
                )
            else:
                st.session_state.attributes[field["key"]] = st.text_input(
                    field["label"],
                    value=existing,
                    key=f"drw_attr_{field['key']}",
                )

    # Custom attributes section
    st.write("---")
    st.write("### Add Custom Attributes")

    # Button to add new custom attribute
    if st.button("➕ Add Custom Attribute"):
        st.session_state.raw_custom_attr_count += 1
        st.rerun()

    # Display custom attribute input fields
    if st.session_state.raw_custom_attr_count > 0:
        st.write("Enter your custom attributes:")

        for i in range(st.session_state.raw_custom_attr_count):
            col_key, col_value, col_remove = st.columns([2, 3, 1])

            with col_key:
                attr_key = st.text_input(
                    f"Attribute Name {i + 1}",
                    key=f"raw_custom_attr_key_{i}",
                    placeholder="e.g., Instrument_Model",
                )

            with col_value:
                attr_value = st.text_input(
                    f"Attribute Value {i + 1}",
                    key=f"raw_custom_attr_value_{i}",
                    placeholder="e.g., Workhorse Sentinel 300",
                )

            with col_remove:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("🗑️", key=f"raw_remove_attr_{i}"):
                    if attr_key in st.session_state.raw_custom_attributes:
                        del st.session_state.raw_custom_attributes[attr_key]
                    st.session_state.raw_custom_attr_count -= 1
                    st.rerun()

            # Store the custom attribute
            if attr_key and attr_value:
                st.session_state.raw_custom_attributes[attr_key] = attr_value

    st.info("Attributes will be added to the NetCDF file once you generate it.")

    # Optionally also copy these attributes onto the working dataset so that
    # later processing pages (e.g. Velocity Processing's magnetic declination
    # lat/lon auto-fill) and the final Write File page can see them.
    st.session_state.copy_attrs_to_processor = st.checkbox(
        "Also copy these attributes to the working dataset "
        "(makes them available on later processing pages, e.g. auto-filling "
        "Latitude/Longitude on the Velocity Processing page)",
        value=st.session_state.get("copy_attrs_to_processor", False),
        key="copy_attrs_to_processor_checkbox",
    )

    if st.session_state.copy_attrs_to_processor:
        if proc is None:
            st.warning(
                "No working dataset available yet — attributes were not copied."
            )
        else:
            merged_attrs = {
                **{k: v for k, v in st.session_state.attributes.items() if v},
                **{
                    k: v
                    for k, v in st.session_state.raw_custom_attributes.items()
                    if v
                },
            }
            # Only re-apply (and log) when the set actually changed, so we
            # don't spam processing_log/config on every rerun.
            if merged_attrs and merged_attrs != st.session_state.get(
                "_synced_raw_attrs"
            ):
                proc.apply_attributes(merged_attrs)
                st.session_state._synced_raw_attrs = dict(merged_attrs)
            if merged_attrs:
                st.caption(
                    f"✓ {len(merged_attrs)} attribute(s) copied to the working dataset."
                )

# File prefix section
st.divider()
st.info(f"Current file name: **{st.session_state.filename}**")

st.session_state.use_custom_filename = st.radio(
    "Do you want to edit Output Filename?",
    ["No", "Yes"],
    horizontal=True,
)

if st.session_state.use_custom_filename == "Yes" and not st.session_state.prefix_saved:
    st.session_state.file_prefix = st.text_input(
        "Enter file name (e.g., GD10A000)",
        value=st.session_state.file_prefix,
    )

    if st.button("Save Filename"):
        if st.session_state.file_prefix.strip():
            st.session_state.prefix_saved = True
            st.rerun()
        else:
            st.warning("Please enter a valid filename before saving.")

if st.session_state.prefix_saved:
    st.success(f"Filename saved as: **{st.session_state.file_prefix}**")

# Axis option
axis_option_DRW: str = st.selectbox(  # type: ignore[assignment]
    "Select x-axis option:",
    options=["time", "ensemble"],
    index=0,
) or "time"
st.session_state.axis_option_DRW = axis_option_DRW

# =============================================================================
# DATA SELECTION CHECKBOXES
# =============================================================================

st.divider()
st.subheader("Select Data Components to Download")

# Entire dataset checkbox
entire_dataset = st.checkbox("📦 **Entire Data Set**", value=False)

st.write("Or select individual components:")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    # If entire dataset is selected, automatically check all boxes
    include_fl = st.checkbox(
        "Fixed Leader",
        value=entire_dataset,
        disabled=entire_dataset,
    )
    include_vl = st.checkbox(
        "Variable Leader",
        value=entire_dataset,
        disabled=entire_dataset,
    )
    include_velocity = st.checkbox(
        "Velocity",
        value=entire_dataset,
        disabled=entire_dataset,
    )

with col2:
    include_echo = st.checkbox(
        "Echo Intensity",
        value=entire_dataset,
        disabled=entire_dataset,
    )
    include_correlation = st.checkbox(
        "Correlation",
        value=entire_dataset,
        disabled=entire_dataset,
    )
    include_pgood = st.checkbox(
        "Percent Good",
        value=entire_dataset,
        disabled=entire_dataset,
    )

# Override individual selections if entire dataset is selected
if entire_dataset:
    include_fl = True
    include_vl = True
    include_velocity = True
    include_echo = True
    include_correlation = True
    include_pgood = True

# Check if any selection is made
any_selected = (
    include_fl
    or include_vl
    or include_velocity
    or include_echo
    or include_correlation
    or include_pgood
)

# Show summary of selection
if any_selected:
    selected_items = []
    if include_fl:
        selected_items.append("Fixed Leader")
    if include_vl:
        selected_items.append("Variable Leader")
    if include_velocity:
        selected_items.append("Velocity")
    if include_echo:
        selected_items.append("Echo Intensity")
    if include_correlation:
        selected_items.append("Correlation")
    if include_pgood:
        selected_items.append("Percent Good")

    st.info(f"**Selected:** {', '.join(selected_items)}")

# Generate button
st.divider()

if st.button("🔄 Generate NetCDF File", type="primary", disabled=not any_selected):
    with st.spinner("Generating NetCDF file..."):
        # Create subset dataset
        subset_ds = create_subset_dataset(
            include_fixed_leader=include_fl,
            include_variable_leader=include_vl,
            include_velocity=include_velocity,
            include_echo=include_echo,
            include_correlation=include_correlation,
            include_percent_good=include_pgood,
        )

        if subset_ds is not None:
            # Write to NetCDF
            filepath = write_netcdf(
                subset_ds,
                axis_option=st.session_state.axis_option_DRW,
                add_attributes=(st.session_state.add_attributes_DRW == "Yes"),
            )

            if filepath:
                st.success("NetCDF file generated successfully!")

                # Show file info
                file_size = os.path.getsize(filepath)
                st.write(f"**File size:** {file_size / 1024:.2f} KB")
                st.write(f"**Variables included:** {len(subset_ds.data_vars)}")

                # Download button
                with open(filepath, "rb") as f:
                    st.download_button(
                        label="⬇️ Download NetCDF File",
                        data=f,
                        file_name=get_prefixed_filename("RAW_DATA.nc"),
                        mime="application/x-netcdf",
                    )

                # Record that the *entire* raw dataset was downloaded as
                # NetCDF, so autoprocess() can reproduce it later - only
                # when it truly is the entire dataset (autoprocess()'s raw
                # export has no component picker, so a partial subset here
                # wouldn't match what it would reproduce).
                if entire_dataset and proc is not None:
                    proc.config.isRawExportOptions = True

# =============================================================================
# CSV DOWNLOAD SECTION
# =============================================================================

st.header("Download Raw Data CSV File", divider="blue")

csv_option: str = st.selectbox(  # type: ignore[assignment]
    "Select data type to download:",
    [
        "Velocity",
        "Echo Intensity",
        "Correlation",
        "Percent Good",
        "Variable Leader",
        "Fixed Leader",
    ],
) or "Velocity"

if csv_option == "Fixed Leader":
    # Combine all Fixed Leader variables
    fl_data = {}
    for var in fl_fields:
        if var in ds.data_vars:
            data = ds[var].values
            # Handle multi-dimensional data by taking first value if needed
            if data.ndim > 1:
                data = data[:, 0] if data.shape[1] > 0 else data.flatten()
            elif data.ndim == 0:
                # Scalar - repeat for all ensembles
                n_ens = ds.sizes.get("time", ds.sizes.get("ensemble", 1))
                data = np.full(n_ens, data.item())
            fl_data[var] = data

    if fl_data:
        download_csv_with_ensemble(fl_data, "Fixed_Leader_All_Variables")
    else:
        st.warning("No Fixed Leader data available for download.")

elif csv_option == "Variable Leader":
    # Combine all Variable Leader variables
    vl_data = {}
    for var in vl_fields:
        if var in ds.data_vars:
            data = ds[var].values
            # Handle multi-dimensional data
            if data.ndim > 1:
                data = data[:, 0] if data.shape[1] > 0 else data.flatten()
            vl_data[var] = data

    if vl_data:
        download_csv(vl_data, "Variable_Leader_All_Variables")
    else:
        st.warning("No Variable Leader data available for download.")

else:
    # Beam data selection
    beam_selection: int = st.radio(  # type: ignore[assignment]
        "Select beam to download",
        (1, 2, 3, 4),
        horizontal=True,
    ) or 1

    # Map selection to data variable
    data_map = {
        "Velocity": "velocity",
        "Echo Intensity": "echo_intensity",
        "Correlation": "correlation",
        "Percent Good": "percent_good",
    }

    var_name = data_map.get(csv_option)

    if var_name and var_name in ds.data_vars:
        data_array = ds[var_name].values

        # Data is typically (beam, cell, time) or (beam, cell, ensemble)
        # Select the beam (0-indexed)
        beam_idx = beam_selection - 1

        if data_array.ndim == 3:
            # Assuming shape is (beam, cell, time)
            download_data = data_array[beam_idx, :, :]
        elif data_array.ndim == 2:
            download_data = data_array
        else:
            st.warning(f"Unexpected data shape: {data_array.shape}")
            download_data = None

        if download_data is not None:
            download_csv_2d(download_data.T, f"{csv_option}_Beam_{beam_selection}")
    else:
        st.warning(f"{csv_option} data not available in dataset.")

# =============================================================================
# SIDEBAR INFO
# =============================================================================

st.sidebar.divider()
st.sidebar.subheader("Dataset Info")

n_vars = len(ds.data_vars)
n_ens = ds.sizes.get("time", ds.sizes.get("ensemble", 0))
n_cells = ds.sizes.get("cell", 0)

st.sidebar.write(f"**Variables:** {n_vars}")
st.sidebar.write(f"**Ensembles:** {n_ens:,}")
st.sidebar.write(f"**Cells:** {n_cells}")

if fl_fields:
    st.sidebar.write(f"**FL Variables:** {len(fl_fields)}")
if vl_fields:
    st.sidebar.write(f"**VL Variables:** {len(vl_fields)}")
