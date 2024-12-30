import numpy as np
import pandas as pd
import tempfile
import math
import os
import plotly.graph_objects as go
import streamlit as st
from plotly_resampler import FigureResampler

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()

# `mask` holds the temporary changes in the page
# `qcmask` holds the final changes in the page
if "mask" not in st.session_state:
    st.session_state.mask = np.copy(st.session_state.orig_mask)

# if not st.session_state.isQCMask:
#     st.write(":grey[Creating a new mask file ...]")
#     st.session_state.qc_mask = np.copy(st.session_state.orig_mask)
#     st.session_state.isSubmit = False
# else:
#     st.write(":grey[Working on a saved mask file ...]")
#     st.write(":orange[WARNING! QC test already completed. Reset to change settings.]")
#     reset_button1 = st.button("Reset Mask Data")
#     if reset_button1:
#         st.session_state.mask = np.copy(st.session_state.orig_mask)
#         st.session_state.qc_mask = np.copy(st.session_state.orig_mask)
#         st.write(":green[Mask data is reset to default]")

if "isThresh" not in st.session_state:
    st.session_state.isThresh = False

# Load data
ds = st.session_state.ds
flobj = st.session_state.flead
vlobj = st.session_state.vlead
velocity = st.session_state.velocity
echo = st.session_state.echo
correlation = st.session_state.correlation
pgood = st.session_state.pgood
ensembles = st.session_state.head.ensembles
cells = flobj.field()["Cells"]
fdata = flobj.fleader
vdata = vlobj.vleader
x = np.arange(0, ensembles, 1)
y = np.arange(0, cells, 1)

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

@st.cache_data
def lineplot(data, title, slope=None, xaxis="time"):
    if xaxis == "time":
        xdata = st.session_state.date
    else:
        xdata = st.session_state.ensemble_axis
    scatter_trace = FigureResampler(go.Figure())
    scatter_trace = go.Scatter(
                                x=xdata,
                                y=data,
                                mode='lines',
                                name=title,
                                marker=dict(color='blue', size=10)  # Customize marker color and size
                                )
    # Create the slope line trace
    if slope is not None:
        line_trace = go.Scatter(
                                x=xdata,
                                y=slope,
                                mode='lines',
                                name='Slope Line',
                                line=dict(color='red', width=2, dash='dash')  # Dashed red line
                                )
        fig = go.Figure(data=[scatter_trace, line_trace])
    else:
        fig = go.Figure(data=[scatter_trace])


    st.plotly_chart(fig)


############## SENSOR HEALTH ######################
st.header("Sensor Health", divider="blue")
st.write("The following details can be used to determine whether the additional sensors are functioning properly.")

# sensor_option = st.selectbox(
#     "Select a data type", ("Pressure Sensor", 
#                            "Temperature Sensor", 
#                            "Heading Sensor", 
#                            "Tilt Sensor: Roll",
#                            "Tilt Sensor: Pitch"))
# ################## Pressure Sensor Check ###################
# if sensor_option == "Pressure Sensor":
################## Fix Orientation ###################

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Pressure", "Salinity", "Temperature", "Heading", "Roll", "Pitch"])
# Sensors
with tab1:
    st.subheader("1. Pressure Sensor Check", divider="orange")
    st.write("""
        Verify whether the pressure sensor is functioning correctly or exhibiting drift. 
        The actual deployment depth can be cross-checked using the mooring diagram 
        for confirmation. To remove outliers, apply the standard deviation method.
    """)
    depth = ds.variableleader.depth_of_transducer
    depth_data = depth.data * depth.scale * 1.0

    leftd, rightd = st.columns([1, 1])
## Clean up the deployment and recovery data
# Compute mean and standard deviation
    depth_median = np.median(depth_data)
    depth_std = np.nanstd(depth_data)
# Get the number of standard deviation
    with rightd:
        depth_no_std = st.number_input("Standard Deviation Cutoff", 0.01, 10.0, 3.0, 0.1)
        depth_xbutton = st.radio("Select an x-axis to plot", ["time", "ensemble"], horizontal=True)
        depth_reset = st.button("Reset Depth to Default")
        if depth_reset:
            st.session_state.depth = st.session_state.vlead.depth_of_transducer.data
            st.session_state.isDepthModified = False
# Mark data above 3 standard deviation as bad
    depth_bad = np.abs(depth_data - depth_median) > depth_no_std*depth_std
    depth_data[depth_bad] = np.nan
    depth_nan = ~np.isnan(depth_data)
# Remove data that are bad
    depth_x = ds.variableleader.rdi_ensemble.data[depth_nan]
    depth_y = depth_data[depth_nan]

## Compute the slope
    depth_slope, depth_intercept = np.polyfit(depth_x, depth_y, 1)
    depth_fitted_line = depth_slope*st.session_state.ensemble_axis + depth_intercept
    depth_change = depth_fitted_line[-1]- depth_fitted_line[0] 

# Display median and slope
    with leftd:
        st.write(":blue-background[Additional Information:]")
        st.write("**Depth Sensor**: ", st.session_state.flead.ez_sensor()["Depth Sensor"])
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")
        st.write(f"**Median depth**: `{depth_median/10} (m)`")
        st.write(f"**Change in depth**: `{np.round(depth_change, 3)} (m)`")
        st.write("**Depth Modified**: ", st.session_state.isDepthModified)

# Plot the data
    # label= depth.long_name + ' (' + depth.unit + ')'
    label= depth.long_name + ' (m)'
    lineplot(depth_data/10, label, slope=depth_fitted_line/10, xaxis=depth_xbutton)

    st.info("""
            If the pressure sensor is not working, upload corrected *CSV* file containing
            the transducer depth.
            The number of ensembles should match the original file.
            The *CSV* file should contain only single column without header.
            """,
            icon="ℹ️"
            )


    depthoption = st.radio("Select method for depth correction:", ["File Upload", "Fixed Value"], horizontal=True)

    if depthoption == "Fixed Value":
        depthinput = st.number_input("Enter corrected depth (m): ", value=None, min_value=0, placeholder="Type a number ...")
        depthbutton = st.button('Change Depth')
        if depthbutton:
            st.session_state.depth = depth_data*0 + int(depthinput*10)
            st.success(f"Depth changed to {depthinput}")
    else:
        uploaded_file_depth = st.file_uploader("Upload Corrected Transducer Depth File", type="csv")
        if uploaded_file_depth is not None:
            st.session_state.pspath = file_access(uploaded_file_depth)
            df_depth = pd.read_csv(st.session_state.pspath, header=None)
            df_numpy_depth = df_depth.to_numpy()
            if len(df_numpy_depth) != st.session_state.head.ensembles:
                st.error(f"""
                        **ERROR: Ensembles not matching.** \\
                        \\
                        Uploaded file ensemble size is {len(df_numpy_depth)}.
                        Actual ensemble size is {st.session_state.head.ensembles}.
                        """, 
                        icon="🚨")
            else:
                st.session_state.depth = df_numpy_depth
                st.session_state.isDepthModified = True
                st.success(" Depth of the transducer modified.", icon="✅")


with tab2:
    st.subheader("2. Conductivity Sensor Check", divider="orange")
    salt = ds.variableleader.salinity
    salt_data = salt.data * salt.scale*1.0

    lefts, rights = st.columns([1, 1])
## Clean up the deployment and recovery data
# Compute mean and standard deviation
    salt_median = np.nanmedian(salt_data)
    salt_std = np.nanstd(salt_data)

    with rights:
        salt_no_std = st.number_input("Standard Deviation Cutoff for salinity", 0.01, 10.0, 3.0, 0.1)
        salt_xbutton = st.radio("Select an x-axis to plot for salinity", ["time", "ensemble"], horizontal=True)

    salt_bad = np.abs(salt_data - salt_median) > salt_no_std*salt_std
    salt_data[salt_bad] = np.nan
    salt_nan = ~np.isnan(salt_data)

# Remove data that are bad
    salt_x = st.session_state.ensemble_axis[salt_nan]
    salt_y = salt_data[salt_nan]

## Compute the slope
    salt_slope, salt_intercept = np.polyfit(salt_x, salt_y, 1)
    salt_fitted_line = salt_slope*st.session_state.ensemble_axis + salt_intercept
    salt_change = salt_fitted_line[-1]-salt_fitted_line[0] 
    
    with lefts:
        st.write(":blue-background[Additional Information:]")
        st.write("Conductivity Sensor: ", st.session_state.flead.ez_sensor()["Conductivity Sensor"])
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")
        st.write(f"Median salinity: {salt_median} $^o$C")
        st.write(f"Change in salinity: {salt_change} $^o$C")

# Plot the data
    label = salt.long_name
    salt_data = np.round(salt_data)
    salt_fitted_line = np.round(salt_fitted_line)
    lineplot(np.int32(salt_data), label, slope=salt_fitted_line, xaxis=salt_xbutton)

    st.info("""
            If the salinity values are not correct or the sensor is not 
            functioning, change the value or upload a
            corrected *CSV* file containing only the temperature values. 
            The *CSV* file must have a single column without a header, 
            and the number of ensembles should match the original file. 
            These updated temperature values will be used to adjust the 
            velocity data and depth cell measurements.
            """,
            icon="ℹ️"
            )

    saltoption = st.radio("Select method", ["Fixed Value", "File Upload"], horizontal=True)

    if saltoption == "Fixed Value":
        saltinput = st.number_input("Enter corrected salinity: ", value=None, min_value=0.0, placeholder="Type a number ...")
        saltbutton = st.button('Change Salinity')
        if saltbutton:
            st.session_state.salt = salt_data*0 + saltinput 
            st.success(f"Salinity changed to {saltinput}")
    else:
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")

        uploaded_file_depth = st.file_uploader("Upload Corrected Salinity File", type="csv")
        if uploaded_file_depth is not None:
            st.session_state.pspath = file_access(uploaded_file_depth)
            df_depth = pd.read_csv(st.session_state.pspath, header=None)
            df_numpy_depth = df_depth.to_numpy()
            if len(df_numpy_depth) != st.session_state.head.ensembles:
                st.error(f"""
                        **ERROR: Ensembles not matching.** \\
                        \\
                        Uploaded file ensemble size is {len(df_numpy_depth)}.
                        Actual ensemble size is {st.session_state.head.ensembles}.
                        """, 
                        icon="🚨")
            else:
                st.session_state.depth = df_numpy_depth
                st.session_state.isSaltModified = True
                st.success("Salinity changed.", icon="✅")

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
        temp_no_std = st.number_input("Standard Deviation Cutoff for Temperature", 0.01, 10.0, 3.0, 0.1)
        temp_xbutton = st.radio("Select an x-axis to plot for temperature", ["time", "ensemble"], horizontal=True)

# Mark data above 3 standard deviation as bad
    temp_bad = np.abs(temp_data - temp_median) > temp_no_std*temp_std
    temp_data[temp_bad] = np.nan
    temp_nan = ~np.isnan(temp_data)
# Remove data that are bad
    temp_x = st.session_state.ensemble_axis[temp_nan]
    temp_y = temp_data[temp_nan]
## Compute the slope
    temp_slope, temp_intercept = np.polyfit(temp_x, temp_y, 1)
    temp_fitted_line = temp_slope*st.session_state.ensemble_axis + temp_intercept
    temp_change = temp_fitted_line[-1]-temp_fitted_line[0] 

    with leftt:
        st.write(":blue-background[Additional Information:]")
        st.write("Temperature Sensor: ", st.session_state.flead.ez_sensor()["Temperature Sensor"])
        st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")
        st.write(f"Median temperature: {temp_median} $^o$C")
        st.write(f"Change in temperature: {np.round(temp_change, 3)} $^o$C")

# Plot the data
    label = temp.long_name + ' (oC)'
    lineplot(temp_data, label, slope=temp_fitted_line, xaxis=temp_xbutton)

# 
    st.info("""
            If the temperature sensor is not functioning, upload a
            corrected *CSV* file containing only the temperature values. 
            The *CSV* file must have a single column without a header, 
            and the number of ensembles should match the original file. 
            These updated temperature values will be used to adjust the 
            velocity data and depth cell measurements.
            """,
            icon="ℹ️"
            )

    st.write(f"Total ensembles: `{st.session_state.head.ensembles}`")

    uploaded_file_temp = st.file_uploader("Upload Corrected Temperature File", type="csv")
    if uploaded_file_temp is not None:
        st.session_state.pspath = file_access(uploaded_file_temp)
        df_temp = pd.read_csv(st.session_state.pspath, header=None)
        df_numpy_temp = df_temp.to_numpy()
        if len(df_numpy_temp) != st.session_state.head.ensembles:
            st.error(f"""
                    **ERROR: Ensembles not matching.** \\
                    \\
                    Uploaded file ensemble size is {len(df_numpy_temp)}.
                    Actual ensemble size is {st.session_state.head.ensembles}.
                    """, 
                    icon="🚨")
        else:
            st.success(" The temperature of transducer modified.", icon="✅")
            st.session_state.temp = df_numpy_temp
            t = df_numpy_temp
            sound_speed = 1449.2 + 4.6*t - 0.055*t**2 
            st.session_state.isTemperatureModified = True


# ################## Heading Sensor Check ###################
with tab4:
    st.warning("""
               WARNING: Heading sensor corrections are currently unavailable. 
               This feature will be included in a future release.
               """, 
               icon="⚠️"
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

    head_xbutton = st.radio("Select an x-axis to plot for headerature", ["time", "ensemble"], horizontal=True)


    st.write(f"Mean heading: {np.round(head_mean_deg, 2)} $^o$")

# Plot the data
    label = head.long_name
    lineplot(head_data, label, xaxis=head_xbutton)

################### Tilt Sensor Check ###################
with tab5:
    st.subheader("4. Tilt Sensor Check: Pitch", divider="orange")
    st.warning("""
               WARNING: Tilt sensor corrections are currently unavailable. 
               This feature will be included in a future release.
               """, 
               icon="⚠️"
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

    pitch_xbutton = st.radio("Select an x-axis to plot for pitcherature", ["time", "ensemble"], horizontal=True)
    st.write(f"Mean pitch: {np.round(pitch_mean_deg, 2)} $^o$")

# Plot the data
    label = pitch.long_name
    lineplot(pitch_data, label, xaxis=pitch_xbutton)

with tab6:
    st.subheader("5. Tilt Sensor Check: Roll", divider="orange")
    st.warning("""
               WARNING: Tilt sensor corrections are currently unavailable. 
               This feature will be included in a future release.
               """, 
               icon="⚠️"
               )
    roll = ds.variableleader.roll
    roll_data = roll.data * roll.scale
# Compute mean
    roll_rad = np.radians(roll_data)
    roll_mean_x = np.mean(np.cos(roll_rad))
    roll_mean_y = np.mean(np.sin(roll_rad))
    roll_mean_rad = np.arctan2(roll_mean_y, roll_mean_x)
    roll_mean_deg = np.degrees(roll_mean_rad)

    roll_xbutton = st.radio("Select an x-axis to plot for rollerature", ["time", "ensemble"], horizontal=True)
    st.write(f"Mean roll: {np.round(roll_mean_deg, 2)} $^o$")

# Plot the data
    label = roll.long_name
    lineplot(roll_data, label, xaxis=roll_xbutton)







