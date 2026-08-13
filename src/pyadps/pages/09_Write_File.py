"""
08_Write_File.py - Write Processed Data Page (Refactored for pyadps v1.0.0)

This page allows users to:
1. Preview processed data (velocity, echo, correlation, percent good)
2. Export processed data to NetCDF or CSV formats
3. Choose between velocity-only export (default) or full dataset export
4. Add custom metadata attributes to exported files
5. Generate configuration files for reproducible processing

Architecture:
- Uses st.session_state.processor (ProcessedDataset) as central state manager
- Uses proc.velocity_to_netcdf() for velocity-only export (default)
- Uses proc.to_netcdf() for full dataset export
- No complex mask variable management - all handled by ProcessedDataset
"""

import json
import os
import tempfile
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Load default attribute definitions from shared config
_ATTR_JSON = os.path.join(os.path.dirname(__file__), "..", "default_attributes.json")
with open(_ATTR_JSON) as _f:
    DEFAULT_ATTRIBUTES = json.load(_f)

# =============================================================================
# PAGE CONFIGURATION AND VALIDATION
# =============================================================================

st.set_page_config(page_title="Write File", page_icon="💾", layout="wide")

# Check if processor exists
if "processor" not in st.session_state or st.session_state.processor is None:
    st.error("⚠️ No data loaded! Please read a file on the **Read File** page first.")
    st.stop()

# Get processor and dataset
proc = st.session_state.processor
ds = proc.dataset


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_total_ensembles() -> int:
    """Get total number of ensembles from dataset."""
    if "time" in ds.dims:
        return ds.sizes["time"]
    elif "ensemble" in ds.dims:
        return ds.sizes["ensemble"]
    return 0


def get_total_cells() -> int:
    """Get total number of cells from dataset."""
    if "cell" in ds.dims:
        return ds.sizes["cell"]
    elif "depth" in ds.dims:
        return ds.sizes["depth"]
    return 0


def get_total_beams() -> int:
    """Get total number of beams from dataset."""
    if "beam" in ds.dims:
        return ds.sizes["beam"]
    return 4


def get_time_axis():
    """Get time axis for plotting."""
    if "time" in ds.coords:
        return pd.to_datetime(ds["time"].values)
    elif "ensemble" in ds.coords:
        return ds["ensemble"].values
    return np.arange(get_total_ensembles())


def get_depth_axis():
    """Get depth/cell axis for plotting."""
    if "depth" in ds.coords:
        return ds["depth"].values
    elif "cell" in ds.coords:
        return ds["cell"].values
    return np.arange(get_total_cells())


def is_earth_coordinates() -> bool:
    """
    Check if data is in Earth coordinates.
    Returns True if transformed to Earth (U, V, W), False for Beam coordinates.
    """
    try:
        coord_info = ds.fixed_leader.coordinate_transformation(ens=0)
        coord_str = coord_info.get("Coordinates", "")
        return "Earth" in coord_str
    except Exception:
        # Fallback: check attrs
        return ds.attrs.get("coordinate_system", "").lower() == "earth"


def get_velocity_labels() -> tuple[str, str, str, str]:
    """Get appropriate velocity component labels based on coordinate system."""
    if is_earth_coordinates():
        return ("U (East)", "V (North)", "W (Vertical)", "Error")
    else:
        return ("Beam 1", "Beam 2", "Beam 3", "Beam 4")


def get_velocity_names() -> dict[str, str]:
    """Resolve the u/v/w variable names from the naming-style selection.

    CF Convention governs attribute values (standard_name, units, etc.), not
    variable names, so all three styles remain CF-compliant — only the
    output variable names differ.
    """
    style = st.session_state.get("velocity_naming_style", "Short (u, v, w)")
    if style == "Short (u, v, w)":
        return {"u": "u", "v": "v", "w": "w"}
    if style == "Custom":
        return {
            "u": st.session_state.get("velocity_name_u") or "u",
            "v": st.session_state.get("velocity_name_v") or "v",
            "w": st.session_state.get("velocity_name_w") or "w",
        }
    return {
        "u": "zonal_velocity",
        "v": "meridional_velocity",
        "w": "vertical_velocity",
    }


def get_file_prefix() -> str:
    """Get file prefix from session state or derive from filename."""
    if "file_prefix" in st.session_state and st.session_state.file_prefix:
        return st.session_state.file_prefix
    elif "fname" in st.session_state and st.session_state.fname:
        return os.path.splitext(os.path.basename(st.session_state.fname))[0]
    return "ADCP"


def get_prefixed_filename(base_name: str) -> str:
    """Generate filename with optional prefix."""
    prefix = get_file_prefix()
    if prefix:
        return f"{prefix}_{base_name}"
    return base_name


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_data_heatmap(
    data: np.ndarray,
    title: str,
    apply_mask: bool = False,
    colorscale: str = "balance",
    missing_value: float = -32768,
) -> None:
    """Plot a 2D heatmap of data (cell x time)."""
    time_axis = get_time_axis()
    depth_axis = get_depth_axis()

    # Handle masking
    plot_data = data.copy().astype(float)

    if apply_mask and "mask" in ds.data_vars:
        mask = ds["mask"].values
        # If mask is 3D (beam, cell, time), use beam 0 for non-velocity data
        if mask.ndim == 3 and plot_data.ndim == 2:
            mask_2d = mask[0, :, :]
        elif mask.ndim == 2:
            mask_2d = mask
        else:
            mask_2d = None

        if mask_2d is not None:
            plot_data = np.where(mask_2d == 1, np.nan, plot_data)

    # Replace missing values with NaN
    plot_data = np.where(plot_data == missing_value, np.nan, plot_data)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=plot_data,
            x=time_axis,
            y=depth_axis,
            colorscale=colorscale,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Depth/Cell",
        height=400,
    )
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)


# Curated diverging colorscales — velocity is centered on zero flow, so only
# scales with a meaningful midpoint are offered (unlike the sequential
# palettes used for Echo/Correlation/Percent Good elsewhere).
DIVERGING_COLORSCALE_OPTIONS = [
    "RdBu_r",
    "RdBu",
    "balance",
    "delta",
    "curl",
    "spectral",
    "picnic",
    "portland",
    "tropic",
    "temps",
    "puor",
    "prgn",
]


def render_diverging_color_scale_options(
    data: np.ndarray, key_suffix: str
) -> tuple[str, float, float]:
    """Render a palette selectbox + symmetric range input for a zero-centered
    diverging heatmap (velocity).

    Unlike a plain min/max range, this uses a single "clamp magnitude" so the
    scale stays properly centered on zero (zmin=-N, zmax=+N) — Plotly ignores
    zmid once zmin/zmax are set explicitly, so symmetry has to be enforced
    here rather than relying on zmid.

    Returns
    -------
    tuple of (colorscale, zmin, zmax)
    """
    data_masked = np.where(data == -32768, np.nan, data)
    if np.all(np.isnan(data_masked)):
        default_range = 1.0
    else:
        default_range = float(
            max(abs(np.nanmin(data_masked)), abs(np.nanmax(data_masked)))
        )

    with st.expander("🎨 Color Scale Options", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            colorscale = st.selectbox(
                "Color palette",
                DIVERGING_COLORSCALE_OPTIONS,
                index=0,
                help="Diverging colorscale centered on zero flow.",
                key=f"colorscale_select_{key_suffix}",
            )
        with col_b:
            clamp_range = st.number_input(
                "Clamp range (± mm/s)",
                min_value=0.0,
                value=default_range,
                help="Values beyond ±this are shown with the colorscale's "
                "end colors, so out-of-range data still gets a color. "
                "Kept symmetric around zero to preserve the diverging scale.",
                key=f"clamp_range_input_{key_suffix}",
            )

    colorscale = colorscale or DIVERGING_COLORSCALE_OPTIONS[0]
    return colorscale, -clamp_range, clamp_range


def plot_velocity_component(
    beam_idx: int,
    title: str,
    apply_mask: bool = False,
    colorscale: str = "RdBu_r",
    zmin: float = None,
    zmax: float = None,
) -> None:
    """Plot velocity component as a heatmap."""
    if "velocity" not in ds.data_vars:
        st.warning("No velocity data available.")
        return

    velocity = ds["velocity"].values
    vel_data = velocity[beam_idx, :, :].copy().astype(float)

    # Apply mask if requested
    if apply_mask and "mask" in ds.data_vars:
        mask = ds["mask"].values
        if mask.ndim == 3:
            vel_data = np.where(mask[beam_idx, :, :] == 1, np.nan, vel_data)

    # Replace missing values
    vel_data = np.where(vel_data == -32768, np.nan, vel_data)

    time_axis = get_time_axis()
    depth_axis = get_depth_axis()

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=vel_data,
            x=time_axis,
            y=depth_axis,
            colorscale=colorscale,
            # zmid only takes effect when zauto is True (i.e. zmin/zmax are
            # None) — once an explicit symmetric range is passed, that
            # already centers the scale on zero without zmid.
            zmid=0 if zmin is None and zmax is None else None,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="mm/s"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Depth/Cell",
        height=400,
    )
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize page-specific session state
if "write_initialized" not in st.session_state:
    st.session_state.write_initialized = False

if not st.session_state.write_initialized:
    # File settings
    st.session_state.file_prefix = get_file_prefix()

    # Export options
    st.session_state.export_format = "NetCDF"
    st.session_state.apply_mask_export = True
    st.session_state.velocity_units = "cm/s"

    # Data component selection (mirrors Download Raw File page)
    st.session_state.export_entire_dataset = False
    st.session_state.export_include_velocity = True  # Default: velocity only
    st.session_state.export_include_echo = False
    st.session_state.export_include_correlation = False
    st.session_state.export_include_percent_good = False
    st.session_state.export_include_mask = False

    # Velocity variable naming
    st.session_state.velocity_naming_style = "Short (u, v, w)"
    st.session_state.velocity_name_u = "zonal_velocity"
    st.session_state.velocity_name_v = "meridional_velocity"
    st.session_state.velocity_name_w = "vertical_velocity"

    # Standard attributes: seeded from Page 03 values if the user filled them in
    st.session_state.add_attributes = False
    raw_attrs = st.session_state.get("attributes", {})
    st.session_state.write_std_attributes = {
        field["key"]: str(raw_attrs[field["key"]]) if field["key"] in raw_attrs and raw_attrs[field["key"]] else ""
        for field in DEFAULT_ATTRIBUTES
    }

    # Extra custom attributes added on this page
    st.session_state.write_custom_attributes = {}
    st.session_state.write_custom_attr_count = 0

    st.session_state.write_initialized = True


# =============================================================================
# PAGE HEADER
# =============================================================================

_, _col_fname = st.columns([5, 1])
with _col_fname:
    st.caption(f"📂 {st.session_state.get('fname', 'No file selected')}")
st.header("💾 Write Processed Data", divider="blue")

st.write("""
Export your processed ADCP data to NetCDF or CSV format. Choose which
components to include — **Velocity** (recommended, selected by default),
**Echo Intensity**, **Correlation**, **Percent Good** — or select
**Entire Dataset** to export everything, including all variables and metadata.
""")


# =============================================================================
# ATTRIBUTE TAB CALLBACKS
# Using on_click callbacks avoids calling st.rerun() explicitly, which would
# reset the active tab back to the first one.
# =============================================================================


def _add_write_custom_attr():
    st.session_state.write_custom_attr_count += 1


def _remove_write_custom_attr(idx: int):
    key_to_remove = st.session_state.get(f"wf_custom_attr_key_{idx}", "")
    if key_to_remove in st.session_state.write_custom_attributes:
        del st.session_state.write_custom_attributes[key_to_remove]
    st.session_state.write_custom_attr_count -= 1


def _remove_raw_custom_attr(attr_key: str):
    st.session_state.raw_custom_attributes.pop(attr_key, None)


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Preview Data", "📝 Attributes", "💾 Export Data", "⚙️ Config File"]
)


# =============================================================================
# TAB 1: PREVIEW DATA
# =============================================================================

with tab1:
    st.header("Preview Processed Data", divider="blue")

    st.write("""
    Preview your processed data before exporting. The mask can be applied to see 
    what the final exported data will look like.
    """)

    # Data type selection
    col1, col2, col3 = st.columns(3)

    with col1:
        var_options = ["Velocity", "Echo Intensity", "Correlation", "Percent Good"]
        var_selection = st.selectbox(
            "Select variable to view", var_options, key="preview_var"
        )

    with col2:
        if var_selection == "Velocity":
            u_label, v_label, w_label, err_label = get_velocity_labels()
            beam_options = [u_label, v_label, w_label, err_label]
            beam_selection = st.selectbox(
                "Select component", beam_options, key="preview_beam"
            )
            beam_idx = beam_options.index(beam_selection)
        else:
            beam_idx = st.selectbox("Select beam", [1, 2, 3, 4], key="preview_beam_num")
            beam_idx = beam_idx - 1  # Convert to 0-indexed

    with col3:
        apply_mask_preview = st.radio(
            "Apply mask?", ["Yes", "No"], horizontal=True, key="preview_mask"
        )
        apply_mask = apply_mask_preview == "Yes"

    # Plot preview — renders live (no button needed) so color-scale option
    # changes update the plot immediately instead of requiring a re-click.
    if var_selection == "Velocity":
        if "velocity" in ds.data_vars:
            vel_data = ds["velocity"].values[beam_idx, :, :]
            colorscale, zmin, zmax = render_diverging_color_scale_options(
                vel_data, key_suffix=f"write_velocity_{beam_idx}_{apply_mask}"
            )
        else:
            colorscale, zmin, zmax = "RdBu_r", None, None
        plot_velocity_component(
            beam_idx,
            f"{var_selection} - {beam_selection}",
            apply_mask,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
        )
    else:
        # Get the appropriate data variable
        var_mapping = {
            "Echo Intensity": "echo_intensity",
            "Correlation": "correlation",
            "Percent Good": "percent_good",
        }
        var_name = var_mapping.get(var_selection, "echo_intensity")

        if var_name in ds.data_vars:
            data = ds[var_name].values
            if data.ndim == 3:  # (beam, cell, time)
                data_2d = data[beam_idx, :, :]
            else:
                data_2d = data

            colorscale = "Viridis" if var_selection != "Correlation" else "Plasma"
            plot_data_heatmap(
                data_2d,
                f"{var_selection} - Beam {beam_idx + 1}",
                apply_mask=apply_mask,
                colorscale=colorscale,
                missing_value=0,
            )
        else:
            st.warning(f"Variable '{var_name}' not found in dataset.")

    # Show processing summary
    with st.expander("📊 Processing Summary", expanded=True):
        stats = proc.get_current_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cells", f"{stats['total_cells']:,}")
        with col2:
            st.metric(
                "Valid Cells", f"{stats['valid']:,}", delta=f"{stats['valid_pct']:.1f}%"
            )
        with col3:
            st.metric(
                "Masked Cells",
                f"{stats['masked']:,}",
                delta=f"-{stats['masked_pct']:.1f}%",
            )

        if hasattr(proc, "processing_log") and proc.processing_log:
            st.write("**Processing Steps Applied:**")
            for step in proc.processing_log:
                st.write(f"- {step}")


# =============================================================================
# TAB 2: ATTRIBUTES
# =============================================================================

with tab2:
    st.header("Attributes", divider="blue")

    st.write("""
    Add metadata attributes to your exported NetCDF file. Fields pre-filled from
    the **Download Raw File** page can be edited here before export.
    """)

    raw_page_filled = any(
        st.session_state.get("attributes", {}).get(f["key"]) for f in DEFAULT_ATTRIBUTES
    )
    if raw_page_filled:
        st.info("📋 Fields below have been pre-filled from the Download Raw File page. Edit as needed.")

    st.session_state.add_attributes = st.checkbox(
        "Add attributes to export",
        value=st.session_state.add_attributes,
        key="add_attrs_checkbox",
    )

    if st.session_state.add_attributes:
        col1, col2 = st.columns(2)
        col_map = {1: col1, 2: col2}

        for field in DEFAULT_ATTRIBUTES:
            with col_map[field["column"]]:
                current = st.session_state.write_std_attributes.get(field["key"], "")
                if field["widget"] == "date_input":
                    try:
                        current_date = (
                            datetime.strptime(current, "%Y-%m-%d").date()
                            if current
                            else None
                        )
                    except ValueError:
                        current_date = None
                    selected_date = st.date_input(
                        field["label"],
                        value=current_date,
                        key=f"wf_attr_{field['key']}",
                    )
                    # Stored as an ISO string so export/config code keeps treating
                    # attribute values uniformly as strings.
                    st.session_state.write_std_attributes[field["key"]] = (
                        selected_date.isoformat()
                        if isinstance(selected_date, date)
                        else ""
                    )
                elif field["widget"] == "text_area":
                    st.session_state.write_std_attributes[field["key"]] = st.text_area(
                        field["label"],
                        value=current,
                        key=f"wf_attr_{field['key']}",
                    )
                else:
                    st.session_state.write_std_attributes[field["key"]] = st.text_input(
                        field["label"],
                        value=current,
                        key=f"wf_attr_{field['key']}",
                    )

        # ── Custom attributes from Download Raw File page ──────────────────────
        raw_custom = st.session_state.get("raw_custom_attributes", {})
        if raw_custom:
            st.write("---")
            st.write("### Custom Attributes from Download Raw File")
            st.caption("Carried over from page 03. Edit values as needed.")
            for attr_key, attr_val in list(raw_custom.items()):
                col_k, col_v, col_del = st.columns([2, 3, 1])
                with col_k:
                    st.text_input("Name", value=attr_key, disabled=True, key=f"wf_rca_k_{attr_key}")
                with col_v:
                    raw_custom[attr_key] = st.text_input(
                        "Value", value=attr_val, key=f"wf_rca_v_{attr_key}"
                    )
                with col_del:
                    st.write("")
                    st.write("")
                    st.button(
                        "🗑️",
                        key=f"wf_rca_del_{attr_key}",
                        on_click=_remove_raw_custom_attr,
                        args=(attr_key,),
                    )
            st.session_state.raw_custom_attributes = raw_custom

        # ── Additional custom attributes added on this page ────────────────────
        st.write("---")
        st.write("### Add Custom Attributes")

        st.button(
            "➕ Add Custom Attribute",
            key="wf_add_custom_attr",
            on_click=_add_write_custom_attr,
        )

        if st.session_state.write_custom_attr_count > 0:
            st.write("Enter your custom attributes:")
            for i in range(st.session_state.write_custom_attr_count):
                col_key, col_value, col_remove = st.columns([2, 3, 1])

                with col_key:
                    attr_key = st.text_input(
                        f"Attribute Name {i + 1}",
                        key=f"wf_custom_attr_key_{i}",
                        placeholder="e.g., Instrument_Model",
                    )
                with col_value:
                    attr_value = st.text_input(
                        f"Attribute Value {i + 1}",
                        key=f"wf_custom_attr_value_{i}",
                        placeholder="e.g., Workhorse Sentinel 300",
                    )
                with col_remove:
                    st.write("")
                    st.write("")
                    st.button(
                        "🗑️",
                        key=f"wf_remove_attr_{i}",
                        on_click=_remove_write_custom_attr,
                        args=(i,),
                    )

                if attr_key and attr_value:
                    st.session_state.write_custom_attributes[attr_key] = attr_value

        # ── Preview ────────────────────────────────────────────────────────────
        with st.expander("Preview All Attributes"):
            all_attrs = {
                **{k: v for k, v in st.session_state.write_std_attributes.items() if v},
                **{k: v for k, v in st.session_state.get("raw_custom_attributes", {}).items() if v},
                **{k: v for k, v in st.session_state.write_custom_attributes.items() if v},
            }
            if all_attrs:
                attrs_df = pd.DataFrame(
                    list(all_attrs.items()), columns=["Attribute", "Value"]
                )
                st.dataframe(attrs_df, hide_index=True, use_container_width=True)
            else:
                st.info("No attributes entered yet.")

    st.info(
        "💡 After configuring attributes, proceed to the **Export Data** tab to download your files."
    )


# =============================================================================
# TAB 3: EXPORT DATA
# =============================================================================

with tab3:
    st.header("Export Processed Data", divider="blue")

    # File prefix
    st.session_state.file_prefix = st.text_input(
        "File prefix",
        value=st.session_state.file_prefix,
        help="Prefix added to all exported filenames",
    )

    st.divider()

    # Output format
    st.session_state.export_format = st.radio(
        "Output format",
        ["NetCDF", "CSV"],
        key="export_format_radio",
        help="NetCDF is recommended for most use cases",
    )

    if st.session_state.export_format == "CSV":
        st.caption(
            "⚠️ `config.ini` remembers which components you select below, "
            "but not the CSV format itself. Reprocessing this config.ini "
            "later (e.g. via Add-Ons → Auto Processing Tool) will always "
            "produce NetCDF, even though this export is CSV."
        )

    st.divider()

    # -------------------------------------------------------------------
    # DATA COMPONENT SELECTION (mirrors Download Raw File page)
    # -------------------------------------------------------------------
    st.subheader("Select Data Components to Export")

    st.session_state.export_entire_dataset = st.checkbox(
        "📦 **Entire Dataset**",
        value=st.session_state.export_entire_dataset,
        help="Exports the complete processed dataset, including all "
        "variables and metadata (equivalent to the previous 'Full Dataset' "
        "option).",
    )

    st.write("Or select individual components:")

    _entire = st.session_state.export_entire_dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.export_include_velocity = st.checkbox(
            "Velocity",
            value=True if _entire else st.session_state.export_include_velocity,
            disabled=_entire,
        )
        st.session_state.export_include_echo = st.checkbox(
            "Echo Intensity",
            value=True if _entire else st.session_state.export_include_echo,
            disabled=_entire,
        )
    with col2:
        st.session_state.export_include_correlation = st.checkbox(
            "Correlation",
            value=True if _entire else st.session_state.export_include_correlation,
            disabled=_entire,
        )
        st.session_state.export_include_percent_good = st.checkbox(
            "Percent Good",
            value=True if _entire else st.session_state.export_include_percent_good,
            disabled=_entire,
        )
    with col3:
        st.session_state.export_include_mask = st.checkbox(
            "QC Mask",
            value=True if _entire else st.session_state.export_include_mask,
            disabled=_entire,
            help="The raw QC mask (1=invalid, 0=valid) used to flag "
            "velocity cells. Export it alongside the raw Echo Intensity/"
            "Correlation/Percent Good values above to see which cells "
            "were flagged without losing the diagnostic data that "
            "explains why.",
        )

    _any_component_selected = _entire or any(
        [
            st.session_state.export_include_velocity,
            st.session_state.export_include_echo,
            st.session_state.export_include_correlation,
            st.session_state.export_include_percent_good,
            st.session_state.export_include_mask,
        ]
    )
    if not _any_component_selected:
        st.warning("⚠️ Select at least one component to export.")

    st.divider()

    # -------------------------------------------------------------------
    # VELOCITY VARIABLE NAMING
    # Not shown for Entire Dataset: that path uses to_netcdf() directly and
    # doesn't apply naming/unit conversion to any variable.
    # -------------------------------------------------------------------
    if not _entire and st.session_state.export_include_velocity:
        st.subheader("Velocity Variable Naming")
        _naming_options = ["CF-style", "Short (u, v, w)", "Custom"]
        st.session_state.velocity_naming_style = st.radio(
            "Naming style",
            _naming_options,
            index=_naming_options.index(st.session_state.velocity_naming_style),
            key="velocity_naming_style_radio",
            help="CF-style (zonal_velocity, meridional_velocity, "
            "vertical_velocity) matches CF Convention long-form naming. "
            "Short (u, v, w) is easier to type and reference in downstream "
            "analysis. Either way, the CF-compliant standard_name attribute "
            "(e.g. eastward_sea_water_velocity) is always included, so the "
            "file remains CF-compliant regardless of the variable name "
            "chosen — CF Convention governs attribute values, not variable "
            "names.",
        )

        if st.session_state.velocity_naming_style == "Custom":
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                st.session_state.velocity_name_u = st.text_input(
                    "U (zonal/eastward) name",
                    value=st.session_state.velocity_name_u,
                )
            with nc2:
                st.session_state.velocity_name_v = st.text_input(
                    "V (meridional/northward) name",
                    value=st.session_state.velocity_name_v,
                )
            with nc3:
                st.session_state.velocity_name_w = st.text_input(
                    "W (vertical) name",
                    value=st.session_state.velocity_name_w,
                )

        st.divider()

    # -------------------------------------------------------------------
    # EXPORT OPTIONS
    # -------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.apply_mask_export = st.checkbox(
            "Apply QC mask to exported data",
            value=st.session_state.apply_mask_export,
            help="Masked cells become NaN in Velocity only. Echo Intensity/"
            "Correlation/Percent Good are always exported as raw, unmasked "
            "values - the mask is velocity-derived (U/V/W/combined "
            "failures), not indexed the same way as these physical-beam "
            "diagnostics, so applying it to them would silently erase "
            "valid readings. Enable the QC Mask component above to see "
            "which cells were flagged.",
        )

    with col2:
        if not _entire and st.session_state.export_include_velocity:
            st.session_state.velocity_units = st.selectbox(
                "Velocity units",
                ["mm/s", "cm/s", "m/s"],
                index=1,  # Default to cm/s
                key="velocity_units_select",
                help="Original data is in mm/s. Select output units.",
            )

    # Show attributes status
    if st.session_state.add_attributes:
        n_attrs = (
            len([v for v in st.session_state.write_std_attributes.values() if v])
            + len([v for v in st.session_state.get("raw_custom_attributes", {}).values() if v])
            + len([v for v in st.session_state.write_custom_attributes.values() if v])
        )
        st.success(f"✅ {n_attrs} attributes will be included in the export.")
    else:
        st.info(
            "ℹ️ No attributes configured. Configure in the **Attributes** tab if needed."
        )

    st.divider()

    # Export button
    st.write("### Generate Export Files")

    if st.button("🚀 Generate Files", type="primary", key="generate_export"):
        try:
            with st.spinner("Generating export files..."):
                # Create temporary directory
                temp_dir = tempfile.mkdtemp()

                # Merge all attribute sources into one dict for export
                all_export_attrs = {}
                if st.session_state.add_attributes:
                    all_export_attrs.update(
                        {k: v for k, v in st.session_state.write_std_attributes.items() if v}
                    )
                    all_export_attrs.update(
                        {k: v for k, v in st.session_state.get("raw_custom_attributes", {}).items() if v}
                    )
                    all_export_attrs.update(
                        {k: v for k, v in st.session_state.write_custom_attributes.items() if v}
                    )
                    proc.apply_attributes(all_export_attrs)

                _entire = st.session_state.export_entire_dataset
                _inc_velocity = _entire or st.session_state.export_include_velocity
                _inc_echo = _entire or st.session_state.export_include_echo
                _inc_correlation = _entire or st.session_state.export_include_correlation
                _inc_percent_good = _entire or st.session_state.export_include_percent_good
                _inc_mask = _entire or st.session_state.export_include_mask

                if st.session_state.export_format == "NetCDF":
                    if _entire:
                        filename = get_prefixed_filename("PRO.nc")
                        filepath = os.path.join(temp_dir, filename)

                        proc.to_netcdf(filepath)

                        with open(filepath, "rb") as f:
                            file_data = f.read()

                        st.download_button(
                            label="📥 Download Entire Dataset NetCDF",
                            data=file_data,
                            file_name=filename,
                            mime="application/x-netcdf",
                        )

                        st.success(f"✅ Entire dataset file generated: {filename}")

                        if st.session_state.add_attributes:
                            st.write(f"📝 Included {len(all_export_attrs)} attributes")

                    else:
                        filename = get_prefixed_filename("PRO.nc")
                        filepath = os.path.join(temp_dir, filename)

                        proc.export_to_netcdf(
                            filepath,
                            include_velocity=_inc_velocity,
                            include_echo=_inc_echo,
                            include_correlation=_inc_correlation,
                            include_percent_good=_inc_percent_good,
                            include_mask=_inc_mask,
                            apply_mask=st.session_state.apply_mask_export,
                            velocity_units=st.session_state.velocity_units,
                            velocity_names=get_velocity_names(),
                            include_metadata=True,
                        )

                        with open(filepath, "rb") as f:
                            file_data = f.read()

                        st.download_button(
                            label="📥 Download NetCDF",
                            data=file_data,
                            file_name=filename,
                            mime="application/x-netcdf",
                        )

                        st.success(f"✅ Export file generated: {filename}")

                        if st.session_state.add_attributes:
                            st.write(f"📝 Included {len(all_export_attrs)} attributes")

                else:  # CSV format
                    st.write("Generating CSV files...")

                    time_axis = get_time_axis()
                    depth_axis = get_depth_axis()
                    # Only velocity uses mask_arr below - it's velocity-derived
                    # (U/V/W/combined failures), not indexed the same way as
                    # the physical-beam Echo Intensity/Correlation/Percent
                    # Good diagnostics, so it's never applied to those.
                    mask_arr = (
                        ds["mask"].values
                        if (st.session_state.apply_mask_export and "mask" in ds.data_vars)
                        else None
                    )

                    def _export_beam_csvs(var_name: str, display_name: str) -> None:
                        """Generate one CSV per beam for a raw (beam, cell, time) variable."""
                        if var_name not in ds.data_vars:
                            return
                        data = ds[var_name].values.astype(float)
                        for beam_idx in range(data.shape[0]):
                            comp_data = data[beam_idx, :, :].copy()
                            comp_data[comp_data == -32768] = np.nan
                            df = pd.DataFrame(
                                comp_data.T, index=time_axis, columns=depth_axis
                            )
                            csv_data = df.to_csv().encode("utf-8")
                            filename = get_prefixed_filename(
                                f"{display_name}_beam{beam_idx + 1}.csv"
                            )
                            st.download_button(
                                label=f"📥 Download {display_name.replace('_', ' ').title()} Beam {beam_idx + 1} CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv",
                                key=f"csv_{display_name}_{beam_idx}",
                            )

                    # Velocity: u/v/w components (raw mm/s, unit selector applies to NetCDF only)
                    if _inc_velocity and "velocity" in ds.data_vars:
                        velocity = ds["velocity"].values
                        vel_names = get_velocity_names()
                        for beam_idx, comp_key in [(0, "u"), (1, "v"), (2, "w")]:
                            vel_data = velocity[beam_idx, :, :].copy().astype(float)
                            vel_data[vel_data == -32768] = np.nan

                            if mask_arr is not None:
                                vel_data = np.where(
                                    mask_arr[beam_idx, :, :] == 1, np.nan, vel_data
                                )

                            df = pd.DataFrame(
                                vel_data.T,
                                index=time_axis,
                                columns=depth_axis,
                            )

                            csv_data = df.to_csv().encode("utf-8")
                            comp_name = vel_names[comp_key]
                            filename = get_prefixed_filename(f"{comp_name}.csv")

                            st.download_button(
                                label=f"📥 Download {comp_name} CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv",
                                key=f"csv_{comp_key}",
                            )

                    if _inc_echo:
                        _export_beam_csvs("echo_intensity", "echo_intensity")
                    if _inc_correlation:
                        _export_beam_csvs("correlation", "correlation")
                    if _inc_percent_good:
                        _export_beam_csvs("percent_good", "percent_good")

                    # Export mask as CSV
                    if _inc_mask and "mask" in ds.data_vars:
                        mask = ds["mask"].values
                        # Use combined mask (beam 3) if available
                        mask_2d = mask[3, :, :] if mask.shape[0] > 3 else mask[0, :, :]
                        mask_df = pd.DataFrame(
                            mask_2d.T, index=time_axis, columns=depth_axis
                        )
                        mask_csv = mask_df.to_csv().encode("utf-8")

                        st.download_button(
                            label="📥 Download Mask CSV",
                            data=mask_csv,
                            file_name=get_prefixed_filename("mask.csv"),
                            mime="text/csv",
                            key="csv_mask",
                        )

                    # Record what was exported, same as the NetCDF paths
                    # (export_to_netcdf()/velocity_to_netcdf()/to_netcdf()
                    # do this internally; CSV export doesn't go through any
                    # of them, so it has to be stamped here directly) - so
                    # "Generate config.ini" reflects this CSV export rather
                    # than stale/default values.
                    proc.config.isExportOptions = True
                    proc.config.export_include_velocity = _inc_velocity
                    proc.config.export_include_echo = _inc_echo
                    proc.config.export_include_correlation = _inc_correlation
                    proc.config.export_include_percent_good = _inc_percent_good
                    proc.config.export_include_mask = _inc_mask
                    proc.config.export_apply_mask = st.session_state.apply_mask_export
                    # CSV velocity is always raw mm/s - the unit selector
                    # only applies to the NetCDF path (see comment above).
                    proc.config.export_velocity_units = "mm/s"

                    st.success("✅ CSV files generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating files: {e}")
            import traceback

            st.code(traceback.format_exc())


# =============================================================================
# TAB 4: CONFIG FILE GENERATOR
# =============================================================================

with tab4:
    st.header("Configuration File Generator", divider="blue")

    st.write("""
    Generate a configuration file (config.ini) that captures all processing settings.
    This file can be used to reproduce the processing using the pyadps autoprocess function.
    """)

    generate_config = st.checkbox(
        "Generate configuration file",
        value=False,
        key="generate_config_checkbox",
    )

    if generate_config:
        st.info("""
        **Note:** Configuration file generation captures the current processing state.
        The generated config.ini can be used with `pyadps.autoprocess()` for batch processing.
        """)

        if st.button("📄 Generate config.ini", key="gen_config_btn"):
            try:
                if st.session_state.add_attributes:
                    all_export_attrs = {}
                    all_export_attrs.update(
                        {k: v for k, v in st.session_state.write_std_attributes.items() if v}
                    )
                    all_export_attrs.update(
                        {k: v for k, v in st.session_state.get("raw_custom_attributes", {}).items() if v}
                    )
                    all_export_attrs.update(
                        {k: v for k, v in st.session_state.write_custom_attributes.items() if v}
                    )
                    proc.apply_attributes(all_export_attrs)

                config_content = proc.export_config_string()

                with st.expander("Preview config.ini", expanded=True):
                    st.code(config_content, language="ini")

                st.download_button(
                    label="📥 Download config.ini",
                    data=config_content.encode("utf-8"),
                    file_name="config.ini",
                    mime="text/plain",
                )

                st.success("✅ Configuration file generated!")

            except Exception as e:
                st.error(f"❌ Error generating config file: {e}")


# =============================================================================
# SIDEBAR: PROCESSING STATUS
# =============================================================================

with st.sidebar:
    st.header("📊 Export Summary")

    # Current statistics
    stats = proc.get_current_stats()

    st.metric("Total Cells", f"{stats['total_cells']:,}")
    st.metric("Valid Cells", f"{stats['valid']:,}", delta=f"{stats['valid_pct']:.1f}%")
    st.metric(
        "Masked Cells", f"{stats['masked']:,}", delta=f"-{stats['masked_pct']:.1f}%"
    )

    st.write("---")

    # Export settings summary
    st.write("**Export Settings:**")
    st.write(f"- Format: {st.session_state.export_format}")
    if st.session_state.export_entire_dataset:
        st.write("- Components: Entire Dataset")
    else:
        _selected_components = [
            name
            for name, flag in [
                ("Velocity", st.session_state.export_include_velocity),
                ("Echo Intensity", st.session_state.export_include_echo),
                ("Correlation", st.session_state.export_include_correlation),
                ("Percent Good", st.session_state.export_include_percent_good),
                ("QC Mask", st.session_state.export_include_mask),
            ]
            if flag
        ]
        st.write(f"- Components: {', '.join(_selected_components) or 'None selected'}")
    st.write(f"- Apply Mask: {'✅' if st.session_state.apply_mask_export else '❌'}")
    if not st.session_state.export_entire_dataset and st.session_state.export_include_velocity:
        st.write(f"- Velocity Units: {st.session_state.velocity_units}")
        st.write(f"- Velocity Naming: {st.session_state.velocity_naming_style}")
    n_attrs_total = (
        len([v for v in st.session_state.write_std_attributes.values() if v])
        + len([v for v in st.session_state.get("raw_custom_attributes", {}).values() if v])
        + len([v for v in st.session_state.write_custom_attributes.values() if v])
    ) if st.session_state.add_attributes else 0
    st.write(f"- Attrs: {'✅ ' + str(n_attrs_total) if st.session_state.add_attributes else '❌'}")

    st.write("---")

    # Dataset info
    st.write("**Dataset Info:**")
    st.write(f"- Ensembles: {get_total_ensembles():,}")
    st.write(f"- Cells: {get_total_cells()}")
    st.write(f"- Beams: {get_total_beams()}")

    # Check coordinate system
    if is_earth_coordinates():
        st.write("- Coords: Earth (U, V, W)")
    else:
        st.write("- Coords: Beam (1, 2, 3, 4)")

    st.write("---")

    # Processing log
    st.write("**Processing Log:**")
    if hasattr(proc, "processing_log") and proc.processing_log:
        for log_entry in proc.processing_log[-5:]:
            st.write(f"- {log_entry}")
    else:
        st.write("*No processing steps applied yet.*")
