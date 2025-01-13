import numpy as np
import pandas as pd
import tempfile
import os
import plotly.graph_objects as go
import streamlit as st
from plotly_resampler import FigureResampler
from pyadps.utils import sensor_health
from utils.sensor_health import sound_speed_correction, tilt_sensor_check

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()

ds = st.session_state.ds


# ----------------- Functions ---------------


# File Access Function
@st.cache_data()
def file_access(uploaded_file):
    """
    Function creates temporary directory to store the uploaded file.
    The path of the file is returned

    Args:
        uploaded_file (string): Name of the uploaded file

    Returns:
        path (string): Path of the uploaded file
    """
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def status_color_map(value):
    # Define a mapping function for styling
    if value == "True":
        return "background-color: green; color: white"
    elif value == "False":
        return "background-color: red; color: white"


# -------------- Widget Functions -------------


# Depth Tab
def set_button_upload_depth():
    if st.session_state.uploaded_file_depth is not None:
        st.session_state.pspath = file_access(st.session_state.uploaded_file_depth)
        df_depth = pd.read_csv(st.session_state.pspath, header=None)
        numpy_depth = df_depth.to_numpy()
        st.session_state.df_numpy_depth = np.squeeze(numpy_depth)
        if len(st.session_state.df_numpy_depth) != st.session_state.head.ensembles:
            st.session_state.isDepthModified_ST = False
        else:
            st.session_state.depth = st.session_state.df_numpy_depth
            st.session_state.isDepthModified_ST = True


def set_button_depth():
    # st.session_state.depth = st.session_state.depth * 0 + int(
    #     st.session_state.fixeddepth_ST * 10
    # )
    st.session_state.depth = np.full(
        st.session_state.head.ensembles, st.session_state.fixeddepth_ST
    )
    st.session_state.depth *= 10
    st.session_state.isDepthModified_ST = True


def reset_button_depth():
    st.session_state.depth = st.session_state.vlead.depth_of_transducer.data
    st.session_state.isDepthModified_ST = False


# Salinity Tab
def set_button_upload_salinity():
    if st.session_state.uploaded_file_salinity is not None:
        st.session_state.pspath = file_access(st.session_state.uploaded_file_salinity)
        df_salinity = pd.read_csv(st.session_state.pspath, header=None)
        numpy_salinity = df_salinity.to_numpy()
        st.session_state.df_numpy_salinity = np.squeeze(numpy_salinity)
        if len(st.session_state.df_numpy_salinity) != st.session_state.head.ensembles:
            st.session_state.isSalinityModified_ST = False
        else:
            st.session_state.salinity = st.session_state.df_numpy_salinity
            st.session_state.isSalinityModified_ST = True


def set_button_salinity():
    st.session_state.salinity = np.full(
        st.session_state.head.ensembles, st.session_state.fixedsalinity_ST
    )
    st.session_state.isSalinityModified_ST = True


def reset_button_salinity():
    st.session_state.salinity = st.session_state.vlead.salinity.data
    st.session_state.isSalinityModified_ST = False


# Temperature Tab
def set_button_upload_temperature():
    if st.session_state.uploaded_file_temperature is not None:
        st.session_state.pspath = file_access(
            st.session_state.uploaded_file_temperature
        )
        df_temperature = pd.read_csv(st.session_state.pspath, header=None)
        numpy_temperature = df_temperature.to_numpy()
        st.session_state.df_numpy_temperature = np.squeeze(numpy_temperature)
        if (
            len(st.session_state.df_numpy_temperature)
            != st.session_state.head.ensembles
        ):
            st.session_state.isTemperatureModified_ST = False
        else:
            st.session_state.temperature = st.session_state.df_numpy_temperature
            st.session_state.isTemperatureModified_ST = True


def set_button_temperature():
    st.session_state.temperature = np.full(
        st.session_state.head.ensembles, fixedtemperature_ST
    )
    st.session_state.isTemperatureModified_ST = True


def reset_button_temperature():
    st.session_state.temperature = st.session_state.vlead.temperature.data
    st.session_state.isTemperatureModified_ST = False


# Corrections/Threshold Tab
def set_threshold_button():
    if st.session_state.sensor_roll_checkbox:
        rollmask = np.copy(st.session_state.sensor_mask_temp)
        roll = ds.variableleader.roll.data
        updated_rollmask = tilt_sensor_check(
            roll, rollmask, cutoff=st.session_state.roll_cutoff_ST
        )
        st.session_state.sensor_mask_temp = updated_rollmask
        st.session_state.isRollCheck_ST = True

    if st.session_state.sensor_pitch_checkbox:
        pitchmask = np.copy(st.session_state.sensor_mask_temp)
        pitch = ds.variableleader.pitch.data
        updated_pitchmask = tilt_sensor_check(
            pitch, pitchmask, cutoff=st.session_state.pitch_cutoff_ST
        )
        st.session_state.sensor_mask_temp = updated_pitchmask
        st.session_state.isPitchCheck_ST = True

    if (
        st.session_state.sensor_fix_velocity_checkbox
        and not st.session_state.sensor_ischeckbox_disabled
    ):
        sound = st.session_state.sound_speed
        t = st.session_state.temperature
        s = st.session_state.salinity
        d = st.session_state.depth
        velocity = sound_speed_correction(
            st.session_state.velocity_sensor, sound, t, s, d
        )
        st.session_state.velocity_temp = velocity
        st.session_state.isVelocityModifiedSound_ST = True


# Save Tab
def reset_threshold_button():
    st.session_state.isRollCheck_ST = False
    st.session_state.isPitchCheck_ST = False
    st.session_state.isVelocityModifiedSound_ST = False
    st.session_state.sensor_mask_temp = np.copy(st.session_state.orig_mask)
    st.session_state.velocity_temp = np.copy(st.session_state.velocity)


def reset_sensor():
    # Deactivate Global Test
    st.session_state.isSensorTest = False
    # Deactivate Local Tests
    st.session_state.isRollCheck_ST = False
    st.session_state.isPitchCheck_ST = False
    # Deactivate Data Modification Tests
    st.session_state.isDepthModified_ST = False
    st.session_state.isSalinityModified_ST = False
    st.session_state.isTemperatureModified_ST = False
    st.session_state.isVelocityModifiedSound_ST = False

    # Reset Mask Data
    # `sensor_mask_temp` holds and transfers the mask changes between each section
    st.session_state.sensor_mask_temp = np.copy(st.session_state.orig_mask)
    # `sensor_mask` holds the final changes in the page after applying save button
    st.session_state.sensor_mask = np.copy(st.session_state.orig_mask)

    # Reset General Data
    #
    # The sensor test includes changes in ADCP data due to sound speed correction
    st.session_state.depth = st.session_state.vlead.depth_of_transducer.data
    st.session_state.salinity = st.session_state.vlead.salinity.data
    st.session_state.temperature = st.session_state.vlead.temperature.data
    # The `velocity_sensor` holds velocity data for correction
    st.session_state.velocity_temp = np.copy(st.session_state.velocity)
    st.session_state.velocity_sensor = np.copy(st.session_state.velocity)


def save_sensor():
    st.session_state.velocity_sensor = np.copy(st.session_state.velocity_temp)
    st.session_state.sensor_mask = np.copy(st.session_state.sensor_mask_temp)
    st.session_state.isSensorTest = True
    # Deactivate Checks for other pages
    st.session_state.isQCTest = False
    st.session_state.isProfileMask = False
    st.session_state.isGridSave = False
    st.session_state.isVelocityMask = False


# Plot Function
@st.cache_data
def lineplot(data, title, slope=None, xaxis="time"):
    if xaxis == "time":
        xdata = st.session_state.date
    else:
        xdata = st.session_state.ensemble_axis
    scatter_trace = FigureResampler(go.Figure())
    scatter_trace = go.Scatter(
        x=xdata, y=data, mode="lines", name=title, marker=dict(color="blue", size=10)
    )
    # Create the slope line trace
    if slope is not None:
        line_trace = go.Scatter(
            x=xdata,
            y=slope,
            mode="lines",
            name="Slope Line",
            line=dict(color="red", width=2, dash="dash"),
        )
        fig = go.Figure(data=[scatter_trace, line_trace])
    else:
        fig = go.Figure(data=[scatter_trace])

    st.plotly_chart(fig)


# Session States
if not st.session_state.isSensorPageReturn:
    st.write(":grey[Creating a new mask file ...]")
    # Check if any test is carried out using isAnyQCTest().
    # If the page is accessed first time, set all sensor session states
    # to default.
    if st.session_state.isFirstSensorVisit:
        reset_sensor()
        st.session_state.isFirstSensorVisit = False
else:
    # If the page is revisited, warn the user not to change the settings
    # without resetting the mask file.
    # if st.session_state.isSensorPageReturn:
    st.write(":grey[Working on a saved mask file ...]")
    st.write(
        ":orange[WARNING! Sensor test already completed. Reset to change settings.]"
    )
    reset_button_saved_mask = st.button("Reset Mask Data", on_click=reset_sensor)

    if reset_button_saved_mask:
        st.write(":green[Mask data is reset to default]")

# ------------------------------------
# -------------WEB PAGES -------------
# ------------------------------------


# ----------- SENSOR HEALTH ----------
st.header("Sensor Health", divider="blue")
st.write(
    """
    The following details can be used to determine whether the
    additional sensors are functioning properly.
    """
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Pressure",
        "Salinity",
        "Temperature",
        "Heading",
        "Roll",
        "Pitch",
        "Corrections",
        "Save/Reset",
    ]
)

# ################## Pressure Sensor Check ###################
with tab1:
    st.subheader("1. Pressure Sensor Check", divider="orange")
    st.write("""
        Verify whether the pressure sensor is functioning correctly
        or exhibiting drift. The actual deployment depth can be
        cross-checked using the mooring diagram for confirmation.
        To remove outliers, apply the standard deviation method.
    """)
    depth = ds.variableleader.depth_of_transducer
    depth_data = depth.data * depth.scale * 1.0

    leftd, rightd = st.columns([1, 1])
    # Clean up the deployment and recovery data
    # Compute mean and standard deviation
    depth_median = np.median(depth_data)
    depth_std = np.nanstd(depth_data)
    # Get the number of standard deviation
    with rightd:
        depth_no_std = st.number_input(
            "Standard Deviation Cutoff", 0.01, 10.0, 3.0, 0.1
        )
        depth_xbutton = st.radio(
            "Select an x-axis to plot", ["time", "ensemble"], horizontal=True
        )
        # Local Reset
        depth_reset = st.button("Reset Depth to Default", on_click=reset_button_depth)
        if depth_reset:
            st.success("Depth reset to default")

    # Mark data above 3 standard deviation as bad
    depth_bad = np.abs(depth_data - depth_median) > depth_no_std * depth_std
    depth_data[depth_bad] = np.nan
    depth_nan = ~np.isnan(depth_data)
    # Remove data that are bad
    depth_x = ds.variableleader.rdi_ensemble.data[depth_nan]
    depth_y = depth_data[depth_nan]

    # Compute the slope
    depth_slope, depth_intercept = np.polyfit(depth_x, depth_y, 1)
    depth_fitted_line = depth_slope * st.session_state.ensemble_axis + depth_intercept
    depth_change = depth_fitted_line[-1] - depth_fitted_line[0]
    st.session_state.sensor_depth_data = depth_data

    # Display median and slope
    with leftd:
        st.write(":blue-background[Additional Information:]")
        st.write(
            "**Depth Sensor**: ", st.session_state.flead.ez_sensor()["Depth Sensor"]
        )
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")
        st.write(f"**Median depth**: `{depth_median/10} (m)`")
        st.write(f"**Change in depth**: `{np.round(depth_change, 3)} (m)`")
        st.write("**Depth Modified**: ", st.session_state.isDepthModified_ST)

    # Plot the data
    # label= depth.long_name + ' (' + depth.unit + ')'
    label = depth.long_name + " (m)"
    lineplot(depth_data / 10, label, slope=depth_fitted_line / 10, xaxis=depth_xbutton)

    st.info(
        """
            If the pressure sensor is not working, upload corrected *CSV*
            file containing the transducer depth. The number of ensembles
            should match the original file. The *CSV* file should contain
            only single column without header.
            """,
        icon="ℹ️",
    )

    st.session_state.depthoption_ST = st.radio(
        "Select method for depth correction:",
        ["File Upload", "Fixed Value"],
        horizontal=True,
    )

    if st.session_state.depthoption_ST == "Fixed Value":
        st.session_state.fixeddepth_ST = st.number_input(
            "Enter corrected depth (m): ",
            value=None,
            min_value=0,
            placeholder="Type a number ...",
        )
        st.session_state.isFixedDepth_ST = st.button(
            "Change Depth", on_click=set_button_depth
        )
        if st.session_state.isFixedDepth_ST:
            st.success(f"Depth changed to {st.session_state.fixeddepth_ST}")
    else:
        st.session_state.uploaded_file_depth = st.file_uploader(
            "Upload Corrected Depth File",
            type="csv",
        )
        if st.session_state.uploaded_file_depth is not None:
            # Check if the number of ensembles match and call button function
            st.session_state.isUploadDepth_ST = st.button(
                "Check & Save Depth", on_click=set_button_upload_depth
            )
            if st.session_state.isUploadDepth_ST:
                if (
                    len(st.session_state.df_numpy_depth)
                    != st.session_state.head.ensembles
                ):
                    st.error(
                        f"""
                        **ERROR: Ensembles not matching.** \\
                        \\
                        Uploaded file ensemble size is {len(st.session_state.df_numpy_depth)}.
                        Actual ensemble size: {st.session_state.head.ensembles}.
                        """,
                        icon="🚨",
                    )
                else:
                    lineplot(
                        np.squeeze(st.session_state.depth.T),
                        title="Modified Depth",
                    )
                    st.success(" Depth of the transducer modified.", icon="✅")


# ################## Conductivity Sensor Check ###################
with tab2:
    st.subheader("2. Conductivity Sensor Check", divider="orange")
    st.write("""
             Verify whether the salinity sensor is functioning properly 
             or showing signs of drift. If a salinity sensor is unavailable, 
             use a constant value. To eliminate outliers, apply the standard 
             deviation method.
             """)
    salinity = ds.variableleader.salinity
    salinity_data = salinity.data * salinity.scale * 1.0

    lefts, rights = st.columns([1, 1])

    # Clean up the deployment and recovery data
    # Compute mean and standard deviation
    salinity_median = np.nanmedian(salinity_data)
    salinity_std = np.nanstd(salinity_data)

    with rights:
        salinity_no_std = st.number_input(
            "Standard Deviation Cutoff for salinity", 0.01, 10.0, 3.0, 0.1
        )
        salinity_xbutton = st.radio(
            "Select an x-axis to plot for salinity",
            ["time", "ensemble"],
            horizontal=True,
        )
        salinity_reset = st.button(
            "Reset Salinity to Default", on_click=reset_button_salinity
        )
        if salinity_reset:
            st.success("Salinity reset to default")

    salinity_bad = (
        np.abs(salinity_data - salinity_median) > salinity_no_std * salinity_std
    )
    salinity_data[salinity_bad] = np.nan
    salinity_nan = ~np.isnan(salinity_data)

    # Remove data that are bad
    salinity_x = st.session_state.ensemble_axis[salinity_nan]
    salinity_y = salinity_data[salinity_nan]

    ## Compute the slope
    salinity_slope, salinity_intercept = np.polyfit(salinity_x, salinity_y, 1)
    salinity_fitted_line = (
        salinity_slope * st.session_state.ensemble_axis + salinity_intercept
    )
    salinity_change = salinity_fitted_line[-1] - salinity_fitted_line[0]

    st.session_state.sensor_salinity_data = salinity_data

    with lefts:
        st.write(":blue-background[Additional Information:]")
        st.write(
            "Conductivity Sensor: ",
            st.session_state.flead.ez_sensor()["Conductivity Sensor"],
        )
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")
        st.write(f"Median salinity: {salinity_median} $^o$C")
        st.write(f"Change in salinity: {salinity_change} $^o$C")
        st.write("**Salinity Modified**: ", st.session_state.isSalinityModified_ST)

    # Plot the data
    label = salinity.long_name
    salinity_data = np.round(salinity_data)
    salinity_fitted_line = np.round(salinity_fitted_line)
    lineplot(
        np.int32(salinity_data),
        label,
        slope=salinity_fitted_line,
        xaxis=salinity_xbutton,
    )

    st.info(
        """
            If the salinity values are not correct or the sensor is not
            functioning, change the value or upload a
            corrected *CSV* file containing only the salinity values.
            The *CSV* file must have a single column without a header,
            and the number of ensembles should match the original file.
            These updated temperature values will be used to adjust the
            velocity data and depth cell measurements.
            """,
        icon="ℹ️",
    )

    st.session_state.salinityoption_ST = st.radio(
        "Select method", ["Fixed Value", "File Upload"], horizontal=True
    )

    if st.session_state.salinityoption_ST == "Fixed Value":
        st.session_state.fixedsalinity_ST = st.number_input(
            "Enter corrected salinity: ",
            value=None,
            min_value=0.0,
            placeholder="Type a number ...",
        )
        st.session_state.isFixedSalinity_ST = st.button(
            "Change Salinity", on_click=set_button_salinity
        )
        if st.session_state.isFixedSalinity_ST:
            st.success(f"Salinity changed to {st.session_state.fixedsalinity_ST}")
            st.session_state.isSalinityModified_ST = True
    else:
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")

        st.session_state.uploaded_file_salinity = st.file_uploader(
            "Upload Corrected Salinity File",
            type="csv",
        )
        if st.session_state.uploaded_file_salinity is not None:
            st.session_state.isUploadSalinity_ST = st.button(
                "Check & Save Salinity", on_click=set_button_upload_salinity
            )
            if st.session_state.isUploadSalinity_ST:
                if (
                    len(st.session_state.df_numpy_salinity)
                    != st.session_state.head.ensembles
                ):
                    st.session_state.isSalinityModified_ST = False
                    st.error(
                        f"""
                            **ERROR: Ensembles not matching.** \\
                            \\
                            Uploaded file ensemble size is {len(st.session_state.df_numpy_salinity)}.
                            Actual ensemble size is {st.session_state.head.ensembles}.
                            """,
                        icon="🚨",
                    )
                else:
                    st.success("Salinity changed.", icon="✅")
                    lineplot(
                        np.squeeze(st.session_state.df_numpy_salinity.T),
                        title="Modified Salinity",
                    )

# ################## Temperature Sensor Check ###################
with tab3:
    # ################## Temperature Sensor Check ###################
    st.subheader("3. Temperature Sensor Check", divider="orange")
    st.write("""
        Verify whether the temperature sensor is functioning correctly or exhibiting drift. 
        The actual deployment depth can be cross-checked using external data (like CTD cast) 
        for confirmation. To remove outliers, apply the standard deviation method.
    """)
    temp = ds.variableleader.temperature
    temp_data = temp.data * temp.scale

    leftt, rightt = st.columns([1, 1])
    ## Clean up the deployment and recovery data
    # Compute mean and standard deviation
    temp_median = np.nanmedian(temp_data)
    temp_std = np.nanstd(temp_data)
    # Get the number of standard deviation
    with rightt:
        temp_no_std = st.number_input(
            "Standard Deviation Cutoff for Temperature", 0.01, 10.0, 3.0, 0.1
        )
        temp_xbutton = st.radio(
            "Select an x-axis to plot for temperature",
            ["time", "ensemble"],
            horizontal=True,
        )
        temp_reset = st.button(
            "Reset Temperature to Default", on_click=reset_button_temperature
        )
        if temp_reset:
            st.success("Temperature Reset to Default")

    # Mark data above 3 standard deviation as bad
    temp_bad = np.abs(temp_data - temp_median) > temp_no_std * temp_std
    temp_data[temp_bad] = np.nan
    temp_nan = ~np.isnan(temp_data)
    # Remove data that are bad
    temp_x = st.session_state.ensemble_axis[temp_nan]
    temp_y = temp_data[temp_nan]
    ## Compute the slope
    temp_slope, temp_intercept = np.polyfit(temp_x, temp_y, 1)
    temp_fitted_line = temp_slope * st.session_state.ensemble_axis + temp_intercept
    temp_change = temp_fitted_line[-1] - temp_fitted_line[0]

    st.session_state.sensor_temp_data = temp_data

    with leftt:
        st.write(":blue-background[Additional Information:]")
        st.write(
            "Temperature Sensor: ",
            st.session_state.flead.ez_sensor()["Temperature Sensor"],
        )
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")
        st.write(f"Median temperature: {temp_median} $^o$C")
        st.write(f"Change in temperature: {np.round(temp_change, 3)} $^o$C")
        st.write(
            "**Temperature Modified**: ", st.session_state.isTemperatureModified_ST
        )

    # Plot the data
    label = temp.long_name + " (oC)"
    lineplot(temp_data, label, slope=temp_fitted_line, xaxis=temp_xbutton)

    #
    st.info(
        """
            If the temperature sensor is not functioning, upload a
            corrected *CSV* file containing only the temperature values. 
            The *CSV* file must have a single column without a header, 
            and the number of ensembles should match the original file. 
            These updated temperature values will be used to adjust the 
            velocity data and depth cell measurements.
            """,
        icon="ℹ️",
    )

    st.session_state.temperatureoption_ST = st.radio(
        "Select method for temperature correction:",
        ["File Upload", "Fixed Value"],
        horizontal=True,
    )

    if st.session_state.temperatureoption_ST == "Fixed Value":
        fixedtemperature_ST = st.number_input(
            "Enter corrected temperature: ",
            value=None,
            min_value=0.0,
            placeholder="Type a number ...",
        )
        st.session_state.isFixedTemperature_ST = st.button(
            "Change Temperature", on_click=set_button_temperature
        )
        if st.session_state.isFixedTemperature_ST:
            st.success(f"Temperature changed to {fixedtemperature_ST}")
            st.session_state.isTemperatureModified_ST = True

    elif st.session_state.temperatureoption_ST == "File Upload":
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")
        st.session_state.uploaded_file_temperature = st.file_uploader(
            "Upload Corrected Temperature File",
            type="csv",
        )
        if st.session_state.uploaded_file_temperature is not None:
            st.session_state.isUploadTemperature_ST = st.button(
                "Check & Save Temperature", on_click=set_button_upload_temperature
            )

            if st.session_state.isUploadTemperature_ST:
                if (
                    len(st.session_state.df_numpy_temperature)
                    != st.session_state.head.ensembles
                ):
                    st.session_state.isTemperatureModified_ST = False
                    st.error(
                        f"""
                            **ERROR: Ensembles not matching.** \\
                            \\
                            Uploaded file ensemble size is {len(st.session_state.df_numpy_temperature)}.
                            Actual ensemble size is {st.session_state.head.ensembles}.
                            """,
                        icon="🚨",
                    )
                else:
                    st.success(" The temperature of transducer modified.", icon="✅")
                    st.session_state.temperature = st.session_state.df_numpy_temperature
                    st.session_state.isTemperatureModified_ST = True
                    lineplot(
                        np.squeeze(st.session_state.df_numpy_temperature.T),
                        title="Modified Temperature",
                    )


# ################## Heading Sensor Check ###################
with tab4:
    st.warning(
        """
               WARNING: Heading sensor corrections are currently unavailable. 
               This feature will be included in a future release.
               """,
        icon="⚠️",
    )
    st.subheader("3. Heading Sensor Check", divider="orange")
    head = ds.variableleader.heading
    head_data = head.data * head.scale

    # Compute mean
    head_rad = np.radians(head_data)
    head_mean_x = np.mean(np.cos(head_rad))
    head_mean_y = np.mean(np.sin(head_rad))
    head_mean_rad = np.arctan2(head_mean_y, head_mean_x)
    head_mean_deg = np.degrees(head_mean_rad)

    head_xbutton = st.radio(
        "Select an x-axis to plot for headerature",
        ["time", "ensemble"],
        horizontal=True,
    )

    st.write(f"Mean heading: {np.round(head_mean_deg, 2)} $^o$")

    # Plot the data
    label = head.long_name
    lineplot(head_data, label, xaxis=head_xbutton)

################### Tilt Sensor Check: Pitch ###################
with tab5:
    st.subheader("4. Tilt Sensor Check: Pitch", divider="orange")
    st.warning(
        """
               WARNING: Tilt sensor corrections are currently unavailable. 
               This feature will be included in a future release.
               """,
        icon="⚠️",
    )

    st.write("The tilt sensor should not show much variation.")

    pitch = ds.variableleader.pitch
    pitch_data = pitch.data * pitch.scale
    # Compute mean
    pitch_rad = np.radians(pitch_data)
    pitch_mean_x = np.mean(np.cos(pitch_rad))
    pitch_mean_y = np.mean(np.sin(pitch_rad))
    pitch_mean_rad = np.arctan2(pitch_mean_y, pitch_mean_x)
    pitch_mean_deg = np.degrees(pitch_mean_rad)

    pitch_xbutton = st.radio(
        "Select an x-axis to plot for pitcherature",
        ["time", "ensemble"],
        horizontal=True,
    )
    st.write(f"Mean pitch: {np.round(pitch_mean_deg, 2)} $^o$")

    # Plot the data
    label = pitch.long_name
    lineplot(pitch_data, label, xaxis=pitch_xbutton)

################### Tilt Sensor Check: Roll ###################
with tab6:
    st.subheader("5. Tilt Sensor Check: Roll", divider="orange")
    st.warning(
        """
               WARNING: Tilt sensor corrections are currently unavailable.
               This feature will be included in a future release.
               """,
        icon="⚠️",
    )
    roll = ds.variableleader.roll
    roll_data = roll.data * roll.scale
    # Compute mean
    roll_rad = np.radians(roll_data)
    roll_mean_x = np.mean(np.cos(roll_rad))
    roll_mean_y = np.mean(np.sin(roll_rad))
    roll_mean_rad = np.arctan2(roll_mean_y, roll_mean_x)
    roll_mean_deg = np.degrees(roll_mean_rad)

    roll_xbutton = st.radio(
        "Select an x-axis to plot for rollerature",
        ["time", "ensemble"],
        horizontal=True,
    )
    st.write(f"Mean roll: {np.round(roll_mean_deg, 2)} $^o$")

    # Plot the data
    label = roll.long_name
    lineplot(roll_data, label, xaxis=roll_xbutton)


with tab7:
    st.subheader("Apply Sensor Thresholds/Corrections", divider="orange")
    col1, col2 = st.columns([0.4, 0.6], gap="large")
    with col1:
        st.session_state.roll_cutoff_ST = st.number_input(
            "Enter roll threshold (deg):", min_value=0, max_value=359, value=15
        )
        st.session_state.pitch_cutoff_ST = st.number_input(
            "Enter pitch threshold (deg):", min_value=0, max_value=359, value=15
        )

    with col2:
        st.write("Select Options:")

        with st.form("Select options"):
            if (
                st.session_state.isTemperatureModified_ST
                or st.session_state.isSalinityModified_ST
            ):
                st.session_state.sensor_ischeckbox_disabled = False
            else:
                st.session_state.sensor_ischeckbox_disabled = True
                st.info("No velocity corrections required.")

            st.session_state.sensor_roll_checkbox = st.checkbox("Roll Threshold")
            st.session_state.sensor_pitch_checkbox = st.checkbox("Pitch Threshold")
            st.session_state.sensor_fix_velocity_checkbox = st.checkbox(
                "Fix Velocity", disabled=st.session_state.sensor_ischeckbox_disabled
            )
            # fix_depth_button = st.checkbox(
            #     "Fix Depth Cell Size", disabled=is_checkbox_disabled
            # )

            submitted = st.form_submit_button("Submit", on_click=set_threshold_button)

        if submitted:
            set_threshold_button()
            # Display Threshold Checks
            if st.session_state.sensor_roll_checkbox:
                st.success("Roll Test Applied")
            if st.session_state.sensor_pitch_checkbox:
                st.success("Pitch Test Applied")
            if (
                st.session_state.sensor_fix_velocity_checkbox
                and not st.session_state.sensor_ischeckbox_disabled
            ):
                st.success("Velocity Correction Applied")

        reset_button_threshold = st.button(
            "Reset Corrections", on_click=reset_threshold_button
        )

        if reset_button_threshold:
            st.info("Data reset to defaults")

with tab8:
    ################## Save Button #############
    st.header("Save Data", divider="blue")
    col1, col2 = st.columns([1, 1])
    with col1:
        save_mask_button = st.button(label="Save Mask Data", on_click=save_sensor)
        if save_mask_button:
            st.success("Mask file saved")

            # Table summarizing changes
            changes_summary = pd.DataFrame(
                [
                    [
                        "Depth Modified",
                        "True" if st.session_state.isDepthModified_ST else "False",
                    ],
                    [
                        "Salinity Modified",
                        "True" if st.session_state.isSalinityModified_ST else "False",
                    ],
                    [
                        "Temperature Modified",
                        "True"
                        if st.session_state.isTemperatureModified_ST
                        else "False",
                    ],
                    [
                        "Pitch Test",
                        "True" if st.session_state.isPitchCheck_ST else "False",
                    ],
                    [
                        "Roll Test",
                        "True" if st.session_state.isRollCheck_ST else "False",
                    ],
                    [
                        "Velocity Correction (Sound)",
                        "True"
                        if st.session_state.isVelocityModifiedSound_ST
                        else "False",
                    ],
                ],
                columns=["Test", "Status"],
            )
            # Apply styles using Styler.apply
            styled_table = changes_summary.style.set_properties(
                **{"text-align": "center"}
            )
            styled_table = styled_table.map(status_color_map, subset=["Status"])

            # Display the styled table
            st.write(styled_table.to_html(), unsafe_allow_html=True)

        else:
            st.warning(" WARNING: Mask data not saved", icon="⚠️")
    with col2:
        # Reset local variables
        reset_mask_button = st.button("Reset mask Data", on_click=reset_sensor)
        if reset_mask_button:
            # Global variables reset
            st.session_state.isSensorTest = False
            st.session_state.isQCTest = False
            st.session_state.isGrid = False
            st.session_state.isProfileMask = False
            st.session_state.isVelocityMask = False
            st.success("Mask data is reset to default")
