"""
Streamlit Page 02: View Raw Data

This page displays all raw variables from the ADCP file without any processing.
Uses the pyadps v1.0.0 API with xarray accessors.

Features:
- Primary data visualization (Velocity, Echo, Correlation, Percent Good) with units
- Variable Leader plots (dynamic sensor data) with important variables first
- Fixed Leader plots (static configuration) with expandable view
- Advanced diagnostics (BIT Results, ADC Channels, Error Status Words)
- Cell number y-axis labels for 2D heatmaps
- Units extracted from xarray attributes (from metadata JSON)

Notes:
- Data is raw - no processing has been applied
- Missing data may exist from pre-deployment quality checks
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly_resampler import FigureResampler

# =============================================================================
# SESSION STATE CHECK
# =============================================================================

if "ds" not in st.session_state or st.session_state.ds is None:
    st.write(":red[Please upload an ADCP file on the Read File page first!]")
    st.stop()

# Get the dataset from session state
ds = st.session_state.ds

# =============================================================================
# DATA EXTRACTION (v1.0.0 API)
# =============================================================================

# Get dimensions
n_ensembles = ds.attrs.get("total_ensembles", len(ds.time))
n_cells = ds.sizes.get("cell", 0)
n_beams = ds.sizes.get("beam", 4)

# Create axis arrays
x_ensemble = np.arange(0, n_ensembles, 1)
y_cells = np.arange(1, n_cells + 1, 1)  # Cell numbers start at 1

# Get time axis
time_data = pd.to_datetime(ds.time.values)

# Get primary data arrays
velocity = ds["velocity"].values if "velocity" in ds.data_vars else None
echo = ds["echo_intensity"].values if "echo_intensity" in ds.data_vars else None
correlation = ds["correlation"].values if "correlation" in ds.data_vars else None
pgood = ds["percent_good"].values if "percent_good" in ds.data_vars else None


# =============================================================================
# IMPORTANT FIELDS (only hardcoded list - for prioritized display)
# =============================================================================

# Important Variable Leader fields to show first
IMPORTANT_VL_FIELDS = [
    "heading",
    "pitch",
    "roll",
    "temperature",
    "transducer_depth",
    "sound_speed",
    "salinity",
    "pressure",
    "pressure_variance",
]

# Important Fixed Leader fields to show first
IMPORTANT_FL_FIELDS = [
    "depth_cell_length",
    "blank_after_transmit",
    "pings_per_ensemble",
    "num_cells",
    "num_beams",
    "low_correlation_threshold",
    "error_velocity_maximum",
    "percent_good_minimum",
    "bin_1_distance",
]

# Fields to exclude from general display (shown in Advanced tab)
ADVANCED_VL_FIELDS = [
    "bit_result",
    "error_status_word_1",
    "error_status_word_2",
    "error_status_word_3",
    "error_status_word_4",
]

# ADC channel prefix
ADC_PREFIX = "adc_channel_"

# BIT result prefix
BIT_PREFIX = "bit_"

# ESW prefix
ESW_PREFIX = "esw"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_unit_from_attrs(var_name: str) -> str:
    """
    Get unit string from xarray variable attributes.

    Parameters
    ----------
    var_name : str
        Variable name in dataset

    Returns
    -------
    str
        Unit string, or empty string if not found
    """
    if var_name not in ds.data_vars:
        return ""

    attrs = ds[var_name].attrs
    unit = attrs.get("units", attrs.get("unit", ""))

    # Handle dimensionless
    if unit in ("1", "dimensionless"):
        return ""

    return unit


def get_long_name(var_name: str) -> str:
    """
    Get long_name from xarray variable attributes, or format the var_name.

    Parameters
    ----------
    var_name : str
        Variable name in dataset

    Returns
    -------
    str
        Human-readable name
    """
    if var_name in ds.data_vars:
        long_name = ds[var_name].attrs.get("long_name", "")
        if long_name:
            return long_name

    # Fallback: format variable name
    return var_name.replace("_", " ").title()


def format_display_name(var_name: str) -> str:
    """Format variable name for display with unit if available."""
    unit = get_unit_from_attrs(var_name)
    display = get_long_name(var_name)
    if unit:
        return f"{display} ({unit})"
    return display


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


@st.cache_data
def fillplot_plotly(
    data: np.ndarray,
    colorscale: str = "balance",
    title: str = "Data",
    xaxis: str = "time",
    units: str = "",
    _time_data: Optional[pd.DatetimeIndex] = None,
    _y_cells: Optional[np.ndarray] = None,
) -> None:
    """
    Create a 2D heatmap plot for beam data.

    Parameters
    ----------
    data : np.ndarray
        2D array (cells x ensembles)
    colorscale : str
        Plotly colorscale name
    title : str
        Plot title
    xaxis : str
        Either 'time' or 'ensemble'
    units : str
        Units for the colorbar
    _time_data : pd.DatetimeIndex, optional
        Time data for x-axis (passed with underscore to help caching)
    _y_cells : np.ndarray, optional
        Cell numbers for y-axis
    """
    if xaxis == "time" and _time_data is not None:
        xdata = _time_data
        xlabel = "Time"
    else:
        xdata = np.arange(data.shape[1])
        xlabel = "Ensemble Number"

    # Use cell numbers for y-axis
    if _y_cells is not None:
        ydata = _y_cells
    else:
        ydata = np.arange(1, data.shape[0] + 1)

    fig = FigureResampler(go.Figure())

    # Replace missing values with NaN for proper display
    data_plot = np.where(data == -32768, np.nan, data)

    # Build colorbar title with units
    colorbar_title = title
    if units:
        colorbar_title = f"{title} ({units})"

    fig.add_trace(
        go.Heatmap(
            z=data_plot,
            x=xdata,
            y=ydata,
            colorscale=colorscale,
            hoverongaps=False,
            colorbar=dict(title=colorbar_title),
        )
    )

    fig.update_layout(
        xaxis=dict(
            showline=True,
            mirror=True,
            title=xlabel,
        ),
        yaxis=dict(
            showline=True,
            mirror=True,
            title="Cell Number",
            autorange="reversed",  # Cell 1 at top
        ),
        title_text=f"{title}{' (' + units + ')' if units else ''}",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def lineplot(
    data: np.ndarray,
    title: str,
    xaxis: str = "time",
    units: str = "",
    _time_data: Optional[pd.DatetimeIndex] = None,
) -> None:
    """
    Create a line plot for time series data.

    Parameters
    ----------
    data : np.ndarray
        1D array of values
    title : str
        Plot title and y-axis label
    xaxis : str
        Either 'time' or 'ensemble'
    units : str
        Units for y-axis
    _time_data : pd.DatetimeIndex, optional
        Time data for x-axis
    """
    # Replace missing values
    data_plot = np.where(data == -32768, np.nan, data)

    # Build y-axis label with units
    ylabel = title
    if units:
        ylabel = f"{title} ({units})"

    if xaxis == "time" and _time_data is not None:
        df = pd.DataFrame({"date": _time_data, title: data_plot})
        fig = px.line(df, x="date", y=title, labels={title: ylabel})
        fig.update_xaxes(title="Time")
    else:
        df = pd.DataFrame({"ensemble": np.arange(len(data_plot)), title: data_plot})
        fig = px.line(df, x="ensemble", y=title, labels={title: ylabel})
        fig.update_xaxes(title="Ensemble Number")

    fig.update_yaxes(title=ylabel)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE CONTENT
# =============================================================================

# Introduction
st.header("View Raw Data", divider="orange")
st.write("""
Displays all variables available in the raw file. **No processing has been applied.**
Data might be missing because of the quality-check criteria used before deployment.

Select either `time` or `ensemble` as the x-axis. The y-axis for 2D heatmaps shows 
**Cell Number** (1 = nearest to transducer).
""")

# X-axis selection
xbutton = st.radio(
    "Select x-axis for plots:",
    ["time", "ensemble"],
    horizontal=True,
    help="Choose time for datetime axis or ensemble for sequential numbering",
    index=0,  # Default to time
)

# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Primary Data", "Variable Leader", "Fixed Leader", "Advanced"]
)

# =============================================================================
# TAB 1: PRIMARY DATA (Velocity, Echo, Correlation, Percent Good)
# =============================================================================

with tab1:
    st.header("Velocity, Echo Intensity, Correlation & Percent Good", divider="blue")

    st.write("""
    These are the main ADCP measurements. Select a data type and beam to visualize.
    
    - **Velocity**: Water velocity measurements (mm/s, NaN = missing)
    - **Echo Intensity**: Backscatter signal strength (counts)
    - **Correlation**: Quality of velocity measurement (counts, higher = better)
    - **Percent Good**: Percentage of valid pings in ensemble (%)
    """)

    col1, col2 = st.columns(2)

    # Build options dict with units from attributes
    primary_options = {}
    if velocity is not None:
        units = get_unit_from_attrs("velocity")
        primary_options["Velocity"] = ("velocity", units if units else "mm/s")
    if echo is not None:
        units = get_unit_from_attrs("echo_intensity")
        primary_options["Echo Intensity"] = (
            "echo_intensity",
            units if units else "counts",
        )
    if correlation is not None:
        units = get_unit_from_attrs("correlation")
        primary_options["Correlation"] = ("correlation", units if units else "counts")
    if pgood is not None:
        units = get_unit_from_attrs("percent_good")
        primary_options["Percent Good"] = ("percent_good", units if units else "%")

    with col1:
        if primary_options:
            var_option = st.selectbox(
                "Select data type:",
                list(primary_options.keys()),
                help="Choose which primary data variable to display",
            )
        else:
            st.warning("No primary data available in this file.")
            var_option = None

    with col2:
        beam = st.radio(
            "Select beam:",
            (1, 2, 3, 4),
            horizontal=True,
            help="Beam 1-4 (for 4-beam ADCP)",
            index=0,  # Default to beam 1
        )

    # Beam naming for Earth coordinates (u, v, w, error)
    # Note: This assumes Earth coordinate transformation is applied
    beam_conversion: dict[int, str] = {
        1: "Zonal (u)",
        2: "Meridional (v)",
        3: "Vertical (w)",
        4: "Error",
    }
    # Plot selected data
    if var_option is not None and beam is not None:
        var_name, units = primary_options[var_option]
        data_array = ds[var_name].values if var_name in ds.data_vars else None

        if data_array is not None and xbutton is not None:
            # Select appropriate colorscale
            colorscales: dict[str, str] = {
                "Velocity": "balance",
                "Echo Intensity": "viridis",
                "Correlation": "plasma",
                "Percent Good": "greens",
            }

            coord = ds.fixed_leader.coordinate_transformation(ens=0).get(
                "Coordinates", "N/A"
            )
            if var_option == "Velocity" and coord == "Earth Coordinates":
                fill_title = (
                    f"{var_option} - {beam_conversion.get(beam, f'Beam {beam}')}"
                )
            else:
                fill_title = f"{var_option} - Beam {beam}"

            fillplot_plotly(
                data_array[beam - 1, :, :],
                colorscale=colorscales.get(var_option, "viridis"),
                title=fill_title,
                xaxis=xbutton,
                units=units,
                _time_data=time_data,
                _y_cells=y_cells,
            )

            cell_number = 10
            cell_number = st.selectbox(
                "Select the Cell Number", ds.coords["cell"].values
            )
            lineplot(
                data_array[beam - 1, cell_number, :],
                fill_title,
                xaxis=xbutton,
                units=units,
                _time_data=time_data,
            )

# =============================================================================
# TAB 2: VARIABLE LEADER
# =============================================================================

with tab2:
    st.header("Variable Leader", divider="blue")

    st.write("""
    Variable Leader data contains **dynamic sensor measurements** that change with 
    each ensemble. These include heading, pitch, roll, temperature, pressure, etc.
    
    **Important fields** are shown first. Expand "More Variables" to see all available fields.
    """)

    # Get VL fields from dataset attributes (set by binary_reader.py)
    vl_fields = ds.attrs.get("variable_leader_variables", [])

    # Filter out advanced fields (shown in Advanced tab)
    vl_fields_display = [
        f
        for f in vl_fields
        if f not in ADVANCED_VL_FIELDS
        and not f.startswith(ADC_PREFIX)
        and not f.startswith(BIT_PREFIX)
        and not f.startswith(ESW_PREFIX)
    ]

    if not vl_fields_display:
        st.warning("No Variable Leader data found in dataset.")
    else:
        # Separate into important and other fields
        important_fields = [f for f in IMPORTANT_VL_FIELDS if f in vl_fields_display]
        other_fields = [f for f in vl_fields_display if f not in IMPORTANT_VL_FIELDS]

        # Important fields selector
        st.subheader("Key Sensor Data")

        if important_fields:
            vl_button_important = st.radio(
                "Select a sensor variable to plot:",
                important_fields,
                horizontal=True,
                format_func=lambda x: get_long_name(x),
                index=0,
            )
            if vl_button_important and xbutton is not None:
                units = get_unit_from_attrs(vl_button_important)
                lineplot(
                    ds[vl_button_important].values,
                    get_long_name(vl_button_important),
                    xaxis=xbutton,
                    units=units,
                    _time_data=time_data,
                )

        # Other fields in expander
        if other_fields:
            with st.expander("More Variables", expanded=False):
                vl_button_other = st.radio(
                    "Select additional variable:",
                    sorted(other_fields),
                    horizontal=True,
                    format_func=lambda x: get_long_name(x),
                    key="vl_other_radio",
                )

                if vl_button_other and xbutton is not None:
                    units = get_unit_from_attrs(vl_button_other)
                    lineplot(
                        ds[vl_button_other].values,
                        get_long_name(vl_button_other),
                        xaxis=xbutton,
                        units=units,
                        _time_data=time_data,
                    )

# =============================================================================
# TAB 3: FIXED LEADER
# =============================================================================

with tab3:
    st.header("Fixed Leader", divider="blue")

    st.write("""
    Fixed Leader data contains **static configuration** that should remain constant 
    throughout the deployment. If these values vary significantly, it may indicate 
    file corruption or configuration changes.
    
    **Key settings** are shown first. These values are typically uniform across all ensembles.
    """)

    # Get FL fields from dataset attributes (set by binary_reader.py)
    fl_fields = ds.attrs.get("fixed_leader_variables", [])

    if not fl_fields:
        st.warning("No Fixed Leader data found in dataset.")
    else:
        # Separate into important and other fields
        important_fields = [f for f in IMPORTANT_FL_FIELDS if f in fl_fields]
        other_fields = [f for f in fl_fields if f not in IMPORTANT_FL_FIELDS]

        # Important fields selector
        st.subheader("Key Configuration")

        if important_fields:
            fl_button_important = st.radio(
                "Select a configuration variable to plot:",
                important_fields,
                horizontal=True,
                format_func=lambda x: get_long_name(x),
            )

            if fl_button_important and xbutton is not None:
                units = get_unit_from_attrs(fl_button_important)
                lineplot(
                    ds[fl_button_important].values,
                    get_long_name(fl_button_important),
                    xaxis=xbutton,
                    units=units,
                    _time_data=time_data,
                )

            # Show summary value
            data = ds[fl_button_important].values
            try:
                unique_vals = np.unique(data[~np.isnan(data.astype(float))])
                if len(unique_vals) == 1:
                    st.success(f"✓ Uniform value: {unique_vals[0]}")
                else:
                    st.warning(
                        f"⚠ Non-uniform: {len(unique_vals)} distinct values found"
                    )
            except (ValueError, TypeError):
                # For non-numeric data
                unique_vals = np.unique(data)
                if len(unique_vals) == 1:
                    st.success(f"✓ Uniform value: {unique_vals[0]}")
                else:
                    st.warning(
                        f"⚠ Non-uniform: {len(unique_vals)} distinct values found"
                    )

        # Other fields in expander
        if other_fields:
            with st.expander("More Variables", expanded=False):
                fl_button_other = st.radio(
                    "Select additional configuration:",
                    sorted(other_fields),
                    horizontal=True,
                    format_func=lambda x: get_long_name(x),
                    key="fl_other_radio",
                )

                if fl_button_other and xbutton is not None:
                    units = get_unit_from_attrs(fl_button_other)
                    lineplot(
                        ds[fl_button_other].values,
                        get_long_name(fl_button_other),
                        xaxis=xbutton,
                        units=units,
                        _time_data=time_data,
                    )

# =============================================================================
# TAB 4: ADVANCED
# =============================================================================

with tab4:
    st.header("Advanced Diagnostics", divider="blue")

    st.write("""
    Advanced diagnostic data including Built-In Test results, ADC channels, 
    and Error Status Words. These are primarily used for troubleshooting 
    instrument issues.
    """)

    adv_option = st.selectbox(
        "Select diagnostic data type:",
        (
            "BIT Result",
            "ADC Channel",
            "Error Status Word 1",
            "Error Status Word 2",
            "Error Status Word 3",
            "Error Status Word 4",
        ),
    )

    if adv_option == "BIT Result":
        st.subheader("Built-In Test (BIT) Result", divider="orange")
        st.write("""
        Contains the results of Workhorse ADCP's built-in test functions.
        A zero indicates a successful BIT result. Non-zero values indicate 
        hardware test failures.
        """)

        # Get BIT fields from dataset
        bit_fields = [
            v for v in ds.data_vars if v.startswith(BIT_PREFIX) or v == "bit_result"
        ]

        if bit_fields:
            bit_button = st.radio(
                "Select BIT field to plot:",
                bit_fields,
                horizontal=True,
                format_func=lambda x: get_long_name(x),
            )

            if bit_button and xbutton is not None:
                units = get_unit_from_attrs(bit_button)
                lineplot(
                    ds[bit_button].values,
                    get_long_name(bit_button),
                    xaxis=xbutton,
                    units=units,
                    _time_data=time_data,
                )

            # Show error summary
            data = ds[bit_button].values
            error_count = np.sum(data != 0)
            if error_count == 0:
                st.success("✓ No errors detected")
            else:
                st.warning(f"⚠ {error_count} ensembles with non-zero values")
        else:
            st.info("BIT Result data not available.")

    elif adv_option == "ADC Channel":
        st.subheader("ADC Channel Data", divider="orange")
        st.write("""
        Analog-to-Digital Converter channel readings. These show the raw 
        sensor values before processing.
        """)

        # Get ADC fields from dataset
        adc_fields = [v for v in ds.data_vars if v.startswith(ADC_PREFIX)]

        if adc_fields:
            adc_button = st.radio(
                "Select ADC channel to plot:",
                sorted(adc_fields),
                horizontal=True,
                format_func=lambda x: f"Channel {x.split('_')[-1]}",
            )

            if adc_button and xbutton is not None:
                units = get_unit_from_attrs(adc_button)
                lineplot(
                    ds[adc_button].values,
                    f"ADC {adc_button.split('_')[-1]}",
                    xaxis=xbutton,
                    units=units if units else "counts",
                    _time_data=time_data,
                )
        else:
            st.info("ADC Channel data not available.")

    elif adv_option is not None and adv_option.startswith("Error Status Word"):
        esw_num = int(adv_option[-1])
        st.subheader(f"Error Status Word {esw_num}", divider="orange")

        esw_descriptions = {
            1: "Contains hardware exception and power status flags.",
            2: "Contains pinging status and wakeup event flags.",
            3: "Contains clock error and timing jump flags.",
            4: "Contains power fail and interrupt error flags.",
        }
        st.write(esw_descriptions.get(esw_num, "Error status flags."))

        # Get ESW fields from dataset
        esw_fields = [
            v
            for v in ds.data_vars
            if v.startswith(f"esw{esw_num}_") or v == f"error_status_word_{esw_num}"
        ]

        if esw_fields:
            esw_button = st.radio(
                "Select error flag to plot:",
                esw_fields,
                horizontal=True,
                format_func=lambda x: get_long_name(x),
            )

            if esw_button and xbutton is not None:
                units = get_unit_from_attrs(esw_button)
                lineplot(
                    ds[esw_button].values,
                    get_long_name(esw_button),
                    xaxis=xbutton,
                    units=units,
                    _time_data=time_data,
                )

            # Show error summary
            data = ds[esw_button].values
            event_count = np.sum(data != 0)
            if event_count == 0:
                st.success("✓ No events detected")
            else:
                st.warning(f"⚠ {event_count} ensembles with events")
        else:
            st.info(f"Error Status Word {esw_num} data not available.")

# =============================================================================
# SIDEBAR INFO
# =============================================================================

st.sidebar.divider()
st.sidebar.subheader("Data Summary")
st.sidebar.write(f"**File:** {st.session_state.fname}")
st.sidebar.write(f"**Ensembles:** {n_ensembles:,}")
st.sidebar.write(f"**Cells:** {n_cells}")
st.sidebar.write(f"**Beams:** {n_beams}")

if time_data is not None and len(time_data) > 1:
    st.sidebar.write(f"**Start:** {time_data[0]}")
    st.sidebar.write(f"**End:** {time_data[-1]}")
