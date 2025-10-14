import os
import tempfile
import time
from typing import Any, Dict, Union, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import utils.readrdi as rd
from utils.signal_quality import default_mask


# Set page configuration to wide layout
st.set_page_config(layout="wide")


"""
Streamlit application for ADCP (Acoustic Doppler Current Profiler) binary file processing.
This page loads ADCP binary files and displays File Header and Fixed Leader data. Once 
the file is loaded, the data processing and visualization options will be available
in other tabs.
"""

# Initialize session state variables if they don't exist
if "fname" not in st.session_state:
    st.session_state.fname = "No file selected"

if "rawfilename" not in st.session_state:
    st.session_state.rawfilename = "rawfile.nc"

if "vleadfilename" not in st.session_state:
    st.session_state.vleadfilename = "vlead.nc"


################ Helper Functions #######################
@st.cache_data()
def read_file(uploaded_file) -> None:
    """
    Creates temporary directory to store the uploaded ADCP binary file.
    Reads ADCP binary file. Stores the file path and data in session state.

    Args:
        uploaded_file: The uploaded file object from Streamlit
    """
    # Create a temporary directory and store the file for reading
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Read the file
    ds = rd.ReadFile(path)
    # By default, the ensemble is fixed while reading
    if not ds.isEnsembleEqual:
        ds.fixensemble()

    # Store file path and data in session state
    st.session_state.fpath = path
    st.session_state.ds = ds


def color_bool(val: bool) -> str:
    """
    Returns CSS color formatting based on boolean value.

    Args:
        val: Boolean value to be colorized

    Returns:
        str: CSS color styling (green for True, red for False, orange for non-boolean)
    """
    if isinstance(val, bool):
        color = "green" if val else "red"
    else:
        color = "orange"
    return f"color: {color}"


def color_bool2(val: str) -> str:
    """
    Returns CSS color formatting based on string value.

    Args:
        val: String value to be colorized

    Returns:
        str: CSS color styling (green for "True"/"healthy", red for "False", orange otherwise)
    """
    if val == "True" or val == "Data type is healthy":
        color = "green"
    elif val == "False":
        color = "red"
    else:
        color = "orange"
    return f"color: {color}"


def initialize_session_state() -> None:
    """Initialize all session state variables for different pages of the application."""

    st.session_state.isTimeAxisModified = False
    st.session_state.isSnapTimeAxis = False
    st.session_state.time_snap_frequency = "h"
    st.session_state.time_snap_tolerance = 5
    st.session_state.time_target_minute = 0
    st.session_state.isTimeGapFilled = False

    # Download Raw File page settings
    st.session_state.add_attributes_DRW = "No"
    st.session_state.axis_option_DRW = "time"
    st.session_state.rawnc_download_DRW = False
    st.session_state.vleadnc_download_DRW = False
    st.session_state.rawcsv_option_DRW = "Velocity"
    st.session_state.rawcsv_beam_DRW = 1
    st.session_state.rawcsv_download_DRW = False

    # Sensor Test page settings
    st.session_state.isSensorTest = False
    st.session_state.isFirstSensorVisit = True

    # Depth Correction settings
    st.session_state.isDepthModified_ST = False
    st.session_state.depthoption_ST = "Fixed Value"
    st.session_state.isFixedDepth_ST = False
    st.session_state.fixeddepth_ST = 0
    st.session_state.isUploadDepth_ST = False

    # Salinity Correction settings
    st.session_state.isSalinityModified_ST = False
    st.session_state.salinityoption_ST = "Fixed Value"
    st.session_state.isFixedSalinity_ST = False
    st.session_state.fixedsalinity_ST = 35
    st.session_state.isUploadSalinity_ST = False

    # Temperature Correction settings
    st.session_state.isTemperatureModified_ST = False
    st.session_state.temperatureoption_ST = "Fixed Value"
    st.session_state.isFixedTemperature_ST = False
    st.session_state.fixedtemperature_ST = 0
    st.session_state.isUploadTemperature_ST = False

    # Pitch, Roll, Velocity Correction settings
    st.session_state.isRollCheck_ST = False
    st.session_state.isPitchCheck_ST = False
    st.session_state.isVelocityModifiedSound_ST = False
    st.session_state.roll_cutoff_ST = 359
    st.session_state.pitch_cutoff_ST = 359

    # QC Test page settings
    st.session_state.isQCTest = False
    st.session_state.isFirstQCVisit = True
    st.session_state.isQCCheck_QCT = False
    st.session_state.ct_QCT = 64
    st.session_state.et_QCT = 0
    st.session_state.evt_QCT = 2000
    st.session_state.ft_QCT = 50
    st.session_state.is3beam_QCT = True
    st.session_state.pgt_QCT = 0
    st.session_state.isBeamModified_QCT = False

    # Profile Test page settings
    st.session_state.isProfileTest = False
    st.session_state.isFirstProfileVisit = True
    st.session_state.isTrimEndsCheck_PT = False
    st.session_state.start_ens_PT = 0
    st.session_state.end_ens_PT = st.session_state.head.ensembles
    st.session_state.isCutBinSideLobeCheck_PT = False
    st.session_state.extra_cells_PT = 0
    st.session_state.water_depth_PT = 0
    st.session_state.isCutBinManualCheck_PT = False
    st.session_state.isRegridCheck_PT = False
    st.session_state.end_cell_option_PT = "Cell"
    st.session_state.interpolate_PT = "nearest"
    st.session_state.manualdepth_PT = 0

    # Velocity Test page settings
    st.session_state.isVelocityTest = False
    st.session_state.isFirstVelocityVisit = True
    st.session_state.isMagnetCheck_VT = False
    st.session_state.magnet_method_VT = "pygeomag"
    st.session_state.magnet_lat_VT = 0
    st.session_state.magnet_lon_VT = 0
    st.session_state.magnet_year_VT = 2025
    st.session_state.magnet_depth_VT = 0
    st.session_state.magnet_user_input_VT = 0
    st.session_state.isCutoffCheck_VT = False
    st.session_state.maxuvel_VT = 250
    st.session_state.maxvvel_VT = 250
    st.session_state.maxwvel_VT = 15
    st.session_state.isDespikeCheck_VT = False
    st.session_state.despike_kernel_VT = 5
    st.session_state.despike_cutoff_VT = 3
    st.session_state.isFlatlineCheck_VT = False
    st.session_state.flatline_kernel_VT = 5
    st.session_state.flatline_cutoff_VT = 3

    # Write File page settings
    st.session_state.isWriteFile = True
    st.session_state.isAttributes = False
    st.session_state.mask_data_WF = "Yes"
    st.session_state.file_type_WF = "NetCDF"
    st.session_state.isProcessedNetcdfDownload_WF = True
    st.session_state.isProcessedCSVDownload_WF = False

    # Page return flags
    st.session_state.isSensorPageReturn = False
    st.session_state.isQCPageReturn = False
    st.session_state.isProfilePageReturn = False
    st.session_state.isVelocityPageReturn = False


def display_diagnostic_plots(ds):
    """Displays the interactive plots for diagnosing the time axis."""
    st.subheader("1. Diagnose with a Plot")

    plot_choice = st.selectbox(
        "Select plot type:",
        ("Time Interval Between Ensembles", "Time Components vs. Ensembles"),
    )

    if plot_choice == "Time Interval Between Ensembles":
        time_s = ds.time
        if len(time_s) > 1:
            time_diff_seconds = time_s.diff().dt.total_seconds().dropna()
            plot_df = pd.DataFrame({"Time Difference (seconds)": time_diff_seconds})
            plot_df.index.name = "Ensemble Number"
            plot_df.reset_index(inplace=True)

            fig = px.line(
                plot_df,
                x="Ensemble Number",
                y="Time Difference (seconds)",
                title="Time Interval Between Ensembles",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "**How to interpret this chart:** A **flat horizontal line** indicates a perfectly regular time interval. **Spikes** represent large gaps in the data. A **noisy or wandering line** indicates time drift."
            )
        else:
            st.write("Not enough data points to plot intervals.")

    elif plot_choice == "Time Components vs. Ensembles":
        time_s = ds.time
        if len(time_s) > 0:
            selected_comp = st.radio(
                "Select time component to plot:",
                ("Hour", "Minute", "Second"),
                index=1,
                horizontal=True,
                key="time_comp_plot_radio",
            )

            plot_df = pd.DataFrame(
                {
                    "Ensemble Number": np.arange(len(time_s)),
                    "Hour": time_s.dt.hour,
                    "Minute": time_s.dt.minute,
                    "Second": time_s.dt.second,
                }
            )

            fig = px.line(
                plot_df,
                x="Ensemble Number",
                y=selected_comp,
                title=f"{selected_comp} Component vs. Ensembles",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "**How to interpret this chart:** For regular data, the **Hour** plot should be a repeating sawtooth wave. The **Minute** and **Second** plots should be flat lines at 0. Jumps indicate gaps, while slopes or noise indicate drift."
            )
        else:
            st.write("No data to plot.")


def display_diagnostic_tables(ds):
    """
    Displays the frequency distribution plots by default.
    Provides optional checkboxes to view the detailed data tables.
    """
    st.subheader("2. Diagnose with Frequency Distributions")
    st.markdown(
        "Use the plots for a quick visual check of time component and interval frequencies. Check the boxes to see the detailed data tables."
    )

    st.markdown("##### Component Frequencies")
    comp = st.selectbox(
        "Select time component to analyze:",
        ("minute", "hour", "second"),
        key="time_comp_select",
    )
    if comp and hasattr(ds, "get_time_component_frequency"):
        freq_series = ds.get_time_component_frequency(comp)
        if not freq_series.empty:
            freq_df = freq_series.reset_index()
            freq_df.columns = [comp.capitalize(), "Count"]

            # The plot is now displayed by default
            fig_comp = px.bar(
                freq_df,
                x=comp.capitalize(),
                y="Count",
                title=f"Distribution of '{comp.capitalize()}' values",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # The table is now optional
            if st.checkbox(f"Show data table for '{comp.capitalize()}' frequency"):
                st.dataframe(freq_df, use_container_width=True)
        else:
            st.write("No time data to analyze.")

    st.markdown("---")
    st.markdown("##### Interval Frequencies")
    interval_freq = ds.get_time_interval_frequency()
    if interval_freq:
        df_interval = pd.DataFrame(
            interval_freq.items(), columns=["Time Difference", "Frequency"]
        )

        # The plot is now displayed by default
        fig_interval = px.bar(
            df_interval,
            x="Time Difference",
            y="Frequency",
            title="Distribution of Time Intervals",
        )
        st.plotly_chart(fig_interval, use_container_width=True)

        # The table is now optional
        if st.checkbox("Show data table for interval frequency"):
            st.dataframe(df_interval, use_container_width=True)

    else:
        st.write("Not enough data to calculate intervals.")


def display_correction_tools(ds):
    """Displays the widgets for correcting the time axis."""
    # --- Step 3: Correction for minor drifts (Snap) ---
    st.subheader("3. Correct Minor Time Drifts (Snap)")
    st.markdown(
        "Use this if your data has small, inconsistent time drifts. This will **round** each timestamp to the nearest specified interval or minute."
    )

    col1, col2 = st.columns(2)
    snap_freq = "h"
    target_minute = None

    with col1:
        default_target_minute = 0
        minute_freq = ds.get_time_component_frequency("minute")
        if not minute_freq.empty:
            default_target_minute = minute_freq.index[0]
        target_minute = st.slider(
            "Select minute to round to (0-59):",
            min_value=0,
            max_value=59,
            value=default_target_minute,
            step=1,
        )

    with col2:
        st.markdown("**Set Correction Tolerance:**")
        tolerance_min = st.number_input(
            "Max allowed correction (minutes)",
            min_value=1,
            max_value=60,
            value=5,
            step=1,
            help="The operation will be aborted if any timestamp needs to be changed by more than this amount.",
        )
        snap_tolerance = f"{tolerance_min}min"

    if st.button("Snap Time Axis"):
        if hasattr(ds, "snap_time_axis"):
            with st.spinner("Snapping time axis..."):
                success, message = ds.snap_time_axis(
                    freq=snap_freq,
                    tolerance=snap_tolerance,
                    target_minute=target_minute,
                )
            if success:
                st.session_state.isTimeAxisModified = True
                st.session_state.isSnapTimeAxis = True
                st.session_state.time_snap_frequency = snap_freq
                st.session_state.time_snap_tolerance = snap_tolerance
                st.session_state.time_target_minute = target_minute
                st.success(message)
                time.sleep(2)
                st.rerun()
            else:
                st.error(message)
        else:
            st.error("`snap_time_axis` method not found in `readrdi`.")

    # --- Step 4: Fill Missing Time Gaps ---
    st.subheader("4. Fill Missing Time Gaps")
    st.markdown(
        "Use this **after snapping** (if needed) to make the time axis perfectly uniform. It finds any large time gaps and fills them with masked (missing) data."
    )

    if st.button("Fill Time Axis Gaps"):
        if hasattr(ds, "fill_time_axis"):
            with st.spinner("Filling gaps in time axis..."):
                success, message = ds.fill_time_axis()
            if success:
                st.session_state.isTimeAxisModified = True
                st.session_state.isTimeGapFilled = True
                st.success(message)
                time.sleep(2)
                st.rerun()
            else:
                st.warning(message)
        else:
            st.error("`fill_time_axis` method not found in `readrdi`.")


def process_date() -> None:
    """
    Displays time axis info and, if irregular, provides diagnostic and correction tools.
    """
    ds = st.session_state.ds
    is_regular = ds.isTimeRegular
    interval_info = ds.get_time_interval()

    st.header("Time Axis Information", divider="blue")

    if st.session_state.get("isTimeAxisModified", False):
        st.warning(
            "⚠️ **Warning:** The time axis has been modified from its original state."
        )

    st.write(
        """
        This section analyzes the time intervals between your ADCP ensembles to determine
        if the data was recorded at a regular frequency (e.g., every minute, every hour).
        """
    )

    st.write(f"**Total Ensembles:** {ds.ensembles}")
    st.write(f"**Start Time:** {ds.time.iloc[0]}")
    st.write(f"**End Time:** {ds.time.iloc[-1]}")
    st.write(f"**Total Time:** {ds.time.iloc[-1] - ds.time.iloc[0]}")

    st.subheader("Check Time Irregularities", divider="orange")

    if is_regular:
        st.success("**Equal Time Interval:** The time axis is regular and uniform.")
        st.write(f"**Detected Time Interval:** :green[{interval_info}]")
    else:
        st.error("**Equal Time Interval:** The time axis is irregular.")
        st.write(f"**Most Common Interval (Mode):** :red[{interval_info}]")
        st.warning(
            "Irregularities can be caused by minor time drifts during recording or by missing data (gaps). "
            "Use the tools below to diagnose and correct these issues."
        )
        st.markdown("---")

        if st.checkbox("🔬 **Diagnose & Correct Time Axis**"):
            with st.container(border=True):
                # Call the new helper functions
                display_diagnostic_plots(ds)
                display_diagnostic_tables(ds)
                display_correction_tools(ds)

    # Finalize session state variables
    st.session_state.ensemble_axis = np.arange(0, ds.ensembles, 1)
    st.session_state.axis_option = "time"
    st.session_state.date = ds.time
    st.session_state.date1 = ds.time
    st.session_state.date2 = ds.time
    st.session_state.date3 = ds.time


def display_file_header() -> None:
    """Display file header information and health checks."""
    st.header("File Header", divider="blue")
    st.write(
        """
        Header information is the first item sent by the ADCP. You may check the file size, 
        total ensembles, and available data types. The function also checks if the total bytes 
        and data types are uniform for all ensembles.
        """
    )

    # Create two columns for buttons
    left1, right1 = st.columns(2)

    with left1:
        # File health check button
        if st.button("Check File Health"):
            cf = st.session_state.head.check_file()

            # Check if file is healthy
            if (
                cf["File Size Match"]
                and cf["Byte Uniformity"]
                and cf["Data Type Uniformity"]
            ):
                st.write("Your file appears healthy! :sunglasses:")
            else:
                st.write("Your file appears corrupted! :worried:")

            # Format file size
            cf["File Size (MB)"] = "{:,.2f}".format(cf["File Size (MB)"])

            # Display ensemble count
            st.write(
                f"Total no. of Ensembles: :green[{st.session_state.head.ensembles}]"
            )

            # Display health check results with coloring
            df = pd.DataFrame(cf.items(), columns=pd.array(["Check", "Details"]))
            df = df.astype("str")
            st.write(df.style.map(color_bool2, subset="Details"))

    with right1:
        # Display data types button
        if st.button("Display Data Types"):
            st.write(
                pd.DataFrame(
                    st.session_state.head.data_types(),
                    columns=pd.array(["Available Data Types"]),
                )
            )

    # Display warnings if any
    if st.session_state.ds.isWarning:
        st.write(
            """ 
            Warnings detected while reading. Data sets may still be available for processing.
            Click `Display Warning` to display warnings for each data type.
            """
        )

        if st.button("Display Warnings"):
            df2 = pd.DataFrame(
                st.session_state.ds.warnings.items(),
                columns=pd.array(["Data Type", "Warnings"]),
            )
            st.write(df2.style.map(color_bool2, subset=["Warnings"]))


def display_fixed_leader() -> None:
    """Display fixed leader (static variables) information."""
    st.header("Fixed Leader (Static Variables)", divider="blue")
    st.write(
        """
        Fixed Leader data refers to the non-dynamic WorkHorse ADCP data like hardware information 
        and thresholds. Typically, values remain constant over time. They only change when you 
        change certain commands, although there are occasional exceptions. You can confirm this 
        using the :blue[**Fleader Uniformity Check**]. Click :blue[**Fixed Leader**] to display 
        the values for the first ensemble.
        """
    )

    # Uniformity check button
    if st.button("Fleader Uniformity Check"):
        # Get uniformity check results
        uniformity_results = st.session_state.flead.is_uniform()

        # Display non-uniform variables
        st.write("The following variables are non-uniform:")
        for key, is_uniform in uniformity_results.items():
            if not is_uniform:
                st.markdown(f":blue[**- {key}**]")

        # Display all static variables with color coding
        st.write("Displaying all static variables")
        df = pd.DataFrame(uniformity_results, index=[0]).T
        st.write(df.style.map(color_bool))

    # Display fixed leader data button
    if st.button("Fixed Leader"):
        # Convert all values to uint64 to ensure consistent datatypes
        fl_dict = st.session_state.flead.field().items()
        new_dict = {}
        for key, value in fl_dict:
            new_dict[key] = value.astype(np.uint64)

        # Create and display dataframe
        df = pd.DataFrame(
            {
                "Fields": new_dict.keys(),
                "Values": new_dict.values(),
            }
        )
        st.dataframe(df, use_container_width=True)

    # Create three columns for additional information
    left, center, right = st.columns(3)

    with left:
        # Display system configuration
        st.dataframe(st.session_state.flead.system_configuration())

    with center:
        # Display EZ sensor information
        st.dataframe(st.session_state.flead.ez_sensor())

    with right:
        # Display coordinate transformation with color coding
        df = pd.DataFrame(st.session_state.flead.ex_coord_trans(), index=[0]).T
        df = df.astype("str")
        st.write(df.style.map(color_bool2))


def process_uploaded_file(uploaded_file) -> None:
    """
    Process the uploaded ADCP binary file and store relevant data in session state.

    Args:
        uploaded_file: The uploaded file object from Streamlit
    """
    ds = st.session_state.ds

    # Store data in session state
    st.session_state.fname = uploaded_file.name
    st.session_state.head = ds.fileheader
    st.session_state.flead = ds.fixedleader
    st.session_state.vlead = ds.variableleader
    st.session_state.velocity = ds.velocity.data
    st.session_state.echo = ds.echo.data
    st.session_state.correlation = ds.correlation.data
    st.session_state.pgood = ds.percentgood.data
    st.session_state.beam_direction = ds.fixedleader.system_configuration()[
        "Beam Direction"
    ]

    # Store sensor data with scaling applied where needed
    st.session_state.sound_speed = ds.variableleader.speed_of_sound.data
    st.session_state.depth = ds.variableleader.depth_of_transducer.data
    st.session_state.temperature = (
        ds.variableleader.temperature.data * ds.variableleader.temperature.scale
    )
    st.session_state.salinity = (
        ds.variableleader.salinity.data * ds.variableleader.salinity.scale
    )

    # Create mask for velocity data if not already created
    if "orig_mask" not in st.session_state:
        st.session_state.orig_mask = default_mask(ds)

    # Display confirmation message
    st.write(f"You selected `{st.session_state.fname}`")


def main():
    """Main application function that handles the workflow."""
    # Title and file upload
    st.title("ADCP Data Analysis Tool")

    # File uploader for ADCP binary files
    uploaded_file = st.file_uploader("Upload RDI ADCP Binary File", type="000")

    if uploaded_file is not None and uploaded_file.name != st.session_state.get(
        "processed_filename", None
    ):
        # Read the file
        read_file(uploaded_file)

        # Process the uploaded file
        process_uploaded_file(uploaded_file)

        # Initialize all session state variables
        initialize_session_state()

        # Process time data
        process_date()

        # Display file header section
        display_file_header()

        # Display fixed leader section
        display_fixed_leader()

    elif "flead" in st.session_state:
        # If file was previously uploaded and processed
        st.write(f"You selected `{st.session_state.fname}`")
        st.info("Refresh to upload a new file.")

        process_date()
        # Display file header and fixed leader information
        display_file_header()
        display_fixed_leader()

    else:
        st.info("Please upload a file to begin.")
        # No file loaded, clear cache and stop
        st.cache_data.clear()
        st.cache_resource.clear()
        st.stop()


if __name__ == "__main__":
    main()
