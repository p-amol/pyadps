import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from streamlit.runtime.state import session_state
from utils.profile_test import side_lobe_beam_angle
from utils.signal_quality import default_mask
from utils.velocity_test import (despike, flatline, #magnetic_declination,
                                 velocity_cutoff, wmm2020api, velocity_modifier)

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()

flobj = st.session_state.flead
vlobj = st.session_state.vlead
fdata = flobj.fleader
vdata = vlobj.vleader


####### Initialize Mask File ##############
# Is this the best way?
if st.session_state.isProfileMask or st.session_state.isQCMask:
    st.write(":grey[Working on a saved mask file ...]")
    if st.session_state.isVelocityMask:
        st.write(
            ":orange[Warning: Velocity test already completed. Reset to change settings.]"
        )
        reset_selectbox = st.selectbox(
            "Choose reset option",
            ("Profile Test", "QC Test", "Default"),
            index=None,
            placeholder="Reset mask to ...",
        )
        if reset_selectbox == "Default":
            st.write("Default mask file selected")
            st.session_state.maskd = st.session_state.orig_mask
            st.session_state.dummyvelocity = st.session_state.velocity
        elif reset_selectbox == "QC Test":
            st.write("QC Test mask file selected")
            st.session_state.maskd = st.session_state.qc_mask
            st.session_state.dummyvelocity = st.session_state.velocity
        elif reset_selectbox == "Profile Test":
            st.session_state.maskd = st.session_state.profile_mask
            if st.session_state.isGridSave:
                st.session_state.dummyvelocity = st.session_state.velocity_regrid
            else:
                st.session_state.dummyvelocity = st.session_state.velocity
        else:
            st.session_state.maskd = st.session_state.velocity_mask
            st.session_state.dummyvelocity = st.session_state.velocity
    else:
        if st.session_state.isProfileMask:
            st.session_state.maskd = st.session_state.profile_mask
        elif st.session_state.isQCMask:
            st.session_state.maskd = st.session_state.qc_mask
        else:
            st.session_state.maskd = st.session_state.orig_mask
else:
    st.write(":grey[Creating a new mask file ...]")


if "isCutoff" not in st.session_state:
    st.session_state.isMagnet = False
    st.session_state.isCutoff = False
    st.session_state.isDespike = False
    st.session_state.isFlatline = False

if "maskd" not in st.session_state:
    if st.session_state.isProfileMask:
        st.session_state.maskd = np.copy(st.session_state.profile_mask)
    elif st.session_state.isQCMask:
        st.session_state.maskd = np.copy(st.session_state.qc_mask)
    else:
        st.session_state.maskd = np.copy(st.session_state.orig_mask)

# If data are not regrided use the default one
if st.session_state.isGridSave:
    st.session_state.dummyvelocity = np.copy(st.session_state.velocity_regrid)
else:
    st.session_state.dummyvelocity = np.copy(st.session_state.velocity)

velocity = st.session_state.dummyvelocity

ensembles = st.session_state.head.ensembles
cells = flobj.field()["Cells"]
x = np.arange(0, ensembles, 1)
y = np.arange(0, cells, 1)


########### Introduction ##########
st.header("Velocity Test", divider="orange")

st.write(
    """
The processing in this page apply only to the velocity data.
"""
)

############ Magnetic Declination ##############
#  Commenting the wmm2020 c based model if needed can be implemented.

# * The magnetic declination is obtained from World Magnetic Model 2020 (WMM2020).
# The python wrapper module `wmm2020` is available from this [Link](https://github.com/space-physics/wmm2020). 

st.header("Magnetic Declination", divider="blue")
st.write(
    """

* The API method utilizes the online magnetic declination service provided by the National Geophysical Data Center (NGDC)
of the National Oceanic and Atmospheric Administration (NOAA) to calculate the magnetic declination. The service is available at this [link](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#declination).
Internet connection is necessary for this method to work.

* According to the year, different models are used for calculating magnetic declination.
    * From 2019 till date, WMM2020 (World Magnetic Model) 
    * From 2000 to 2018 EMM (Enhanced Magnetic Model)
    * Before 1999 IGRF (International Geomagnetic Reference Field)

* In the manual method, the user can directly enter the magnetic declination.

If the magnetic declination is reset, re-run the remaining tests again.
"""
)

# Selecting the method to calculate magnetic declination.
# method = st.radio("Select a method", ("WMM2020", "API", "Manual"), horizontal=True)
method = st.radio("Select a method", ("API", "Manual"), horizontal=True)
# method = method - 1
st.session_state.method = method

if "isMagnetButton" not in st.session_state:
    st.session_state.isMagnetButton = False

# Track button clicks
if "isButtonClicked" not in st.session_state:
    st.session_state.isButtonClicked = False


def toggle_btns():
    st.session_state.isMagnetButton = not st.session_state.isMagnetButton
    st.session_state.isButtonClicked = not st.session_state.isButtonClicked


with st.form(key="magnet_form"):
    # if st.session_state.method == "WMM2020":
    #     st.session_state.isMagnet = False
    #     lat = st.number_input("Latitude", -90.0, 90.0, 0.0, step=1.0)
    #     lon = st.number_input("Longitude", 0.0, 360.0, 0.1, step=1.0, format="%.4f")
    #     depth = st.number_input("Depth", 0, 1000, 0, step=1)
    #     year = st.number_input("Year", 1950, 2100, 2024, 1)
    if  st.session_state.method == "API":
        st.session_state.isMagnet = False
        lat = st.number_input("Latitude", -90.0, 90.0, 0.0, step=1.0)
        lon = st.number_input("Longitude", 0.0, 360.0, 0.1, step=1.0, format="%.4f")
        year = st.number_input("Year", 1950, 2024, 2024, 1)
    else:
        st.session_state.isMagnet = False
        mag = [[st.number_input("Declination", -180.0, 180.0, 0.0, 0.1)]]
    
    if st.session_state.method == "Manual":
        button_name = "Accept"
    else:
        button_name = "Compute"

    if st.form_submit_button(
        button_name, on_click=toggle_btns, disabled=st.session_state.isMagnetButton
    ):
        # if st.session_state.method == "WMM2020":
        #     try:
        #         mag = magnetic_declination(lat, lon, depth, year)
        #         st.session_state.dummyvelocity = velocity_modifier(velocity, mag)
        #         st.session_state.lat = lat
        #         st.session_state.lon = lon
        #         st.session_state.magnetic_dec_depth = depth
        #         st.session_state.year = year
        #         st.session_state.angle = np.trunc(mag[0][0])
        #         st.session_state.isMagnet = True
        #         st.session_state.isButtonClicked = True
        #     except:
        #         st.write(":red[Process failed! please use other methods: API or  Manual]")

        if  st.session_state.method == "API":
            try:
                mag = wmm2020api(lat, lon, year)
                st.session_state.dummyvelocity = velocity_modifier(velocity, mag)
                st.session_state.lat = lat
                st.session_state.lon = lon
                st.session_state.year = year
                # st.session_state.angle = np.trunc(mag[0][0])
                st.session_state.angle = np.round(mag[0][0], decimals=3 )
                st.session_state.isMagnet = True
                st.session_state.isButtonClicked = True
            except:
                st.write(":red[Connection error! please check the internet or use manual method]")

        else:
            st.session_state.dummyvelocity = velocity_modifier(velocity, mag)
            st.session_state.angle = np.round(mag[0][0], decimals=3 )
            st.session_state.isMagnet = True
            st.session_state.isButtonClicked = True


if st.session_state.isMagnet:
    st.write(f"Magnetic declination: {st.session_state.angle}\u00b0")
    st.write(":green[Magnetic declination correction applied to velocities]")

if st.button(
    "Reset Magnetic Declination",
    on_click=toggle_btns,
    disabled=not st.session_state.isMagnetButton,
):
    st.session_state.dummyvelocity = np.copy(velocity)
    st.session_state.isMagnet = False
    st.session_state.isButtonClicked = False 

############# Velocity Cutoffs #################
st.header("Velocity Cutoffs", divider="blue")
st.write(
    """
Drop velocities whose magnitude is larger than the threshold.
"""
)
with st.form(key="cutbin_form"):
    maxuvel = st.number_input("Maximum Zonal Velocity Cutoff (cm/s)", 0, 2000, 250, 1)
    maxvvel = st.number_input(
        "Maximum Meridional Velocity Cutoff (cm/s)", 0, 2000, 250, 1
    )
    maxwvel = st.number_input("Maximum Vertical Velocity Cutoff (cm/s)", 0, 2000, 15, 1)
    submit_cutoff = st.form_submit_button(label="Submit")

if submit_cutoff:

    st.session_state.maxuvel = maxuvel
    st.session_state.maxvvel = maxvvel
    st.session_state.maxwvel = maxwvel

    st.session_state.maskd = velocity_cutoff(
        velocity[0, :, :], st.session_state.maskd, cutoff=maxuvel
    )
    st.session_state.maskd = velocity_cutoff(
        velocity[1, :, :], st.session_state.maskd, cutoff=maxvvel
    )
    st.session_state.maskd = velocity_cutoff(
        velocity[2, :, :], st.session_state.maskd, cutoff=maxwvel
    )
    st.session_state.isCutoff = True


if st.session_state.isCutoff:
    st.write(":green[Cutoff Applied]")
    a = {
        "Max. Zonal Velocity": maxuvel,
        "Max. Meridional Velocity": maxvvel,
        "Max. Vertical Velocity": maxwvel,
    }
    st.write(a)


############## DESPIKE DATA #################
st.header("Despike Data", divider="blue")
st.write("""A rolling median filter is applied to remove spikes from the data.
The kernal size determines the number of ensembles (time interval) for the filter window.
The standard deviation specifies the maximum allowable deviation to remove the spike.""")

# time_interval = pd.Timedelta(st.session_state.date[-1] - st.session_state.date[0]).seconds/(3600*st.session_state.head.ensembles)

st.write("Time interval: ", st.session_state.date[1] - st.session_state.date[0])

despike_kernal = st.number_input(
    "Enter Despike Kernal Size for Median Filter", 0, st.session_state.head.ensembles, 5, 1
)



despike_cutoff = st.number_input("Standard Deviation Cutoff for Spike Removal", 0.1, 10.0, 3.0, 0.1)
despike_button = st.button("Despike")
if despike_button:

    st.session_state.despike_kernal = despike_kernal
    st.session_state.despike_cutoff = despike_cutoff

    st.session_state.maskd = despike(
        velocity[0, :, :],
        st.session_state.maskd,
        kernal_size=despike_kernal,
        cutoff=despike_cutoff,
    )
    st.session_state.maskd = despike(
        velocity[1, :, :],
        st.session_state.maskd,
        kernal_size=despike_kernal,
        cutoff=despike_cutoff,
    )
    st.session_state.isDespike = True

if st.session_state.isDespike:
    st.write(":green[Data Despiked]")
    b = {
        "Kernal Size": despike_kernal,
        "Despike Cutoff": despike_cutoff,
    }
    st.write(b)

st.header("Remove Flatline", divider="blue")

st.write("""
Flatline removal detects segments of data where values remain constant over 
a specified interval. The kernel size defines the number of consecutive 
ensembles (time intervals) considered in the check, while the threshold sets 
the maximum allowable variation.
""")

st.write("Time interval: ", st.session_state.date[1] - st.session_state.date[0])

flatline_kernal = st.number_input("Enter Flatline Kernal Size", 0, 100, 13, 1)
flatline_cutoff = st.number_input("Enter Flatline deviation (mm/s)", 0, 100, 1, 1)

flatline_button = st.button("Remove Flatline")


if flatline_button:
    st.session_state.flatline_kernal = flatline_kernal
    st.session_state.flatline_cutoff = flatline_cutoff

    st.session_state.maskd = flatline(
        velocity[0, :, :],
        st.session_state.maskd,
        kernal_size=flatline_kernal,
        cutoff=flatline_cutoff,
    )
    st.session_state.maskd = flatline(
        velocity[1, :, :],
        st.session_state.maskd,
        kernal_size=flatline_kernal,
        cutoff=flatline_cutoff,
    )
    st.session_state.maskd = flatline(
        velocity[2, :, :],
        st.session_state.maskd,
        kernal_size=flatline_kernal,
        cutoff=flatline_cutoff,
    )
    st.session_state.isFlatline = True

if st.session_state.isFlatline:
    st.write(":green[Flatline Removed]")
    b = {
        "Kernal Size": flatline_kernal,
        "Flatline Cutoff": flatline_cutoff,
    }
    st.write(b)


##################### SAVE DATA ###################
st.header("Save & Reset Data", divider="blue")


def reset_data():
    st.session_state.dummyvelocity = np.copy(st.session_state.velocity_regrid)
    st.session_state.isMagnet = False
    st.session_state.isCutoff = False
    st.session_state.isDespike = False
    st.session_state.isFlatline = False
    st.session_state.isButtonClicked = False 


col1, col2 = st.columns([1, 1])
with col1:
    save_button = st.button(label="Save Data")
    if save_button:
        st.session_state.veltest_velocity = np.copy(st.session_state.dummyvelocity)
        st.session_state.velocity_mask = np.copy(st.session_state.maskd)
        st.session_state.isVelocityMask = True
        st.write(":green[Mask data saved]")
                # Status Summary Table
        status_summary = pd.DataFrame(
            [
                ["Magnetic Declination", "True" if st.session_state.isButtonClicked  else "False"],
                ["Velocity Cutoffs", "True" if st.session_state.isCutoff else "False"],
                ["Despike Data", "True" if st.session_state.isDespike else "False"],
                ["Remove Flatline", "True" if st.session_state.isFlatline else "False"],
            ],
            columns=["Test", "Status"],
        )

        # Define a mapping function for styling
        def status_color_map(value):
            if value == "True":
                return "background-color: green; color: white"
            elif value == "False":
                return "background-color: red; color: white"
            else:
                return ""

        # Apply styles using Styler.apply
        styled_table = status_summary.style.set_properties(**{"text-align": "center"})
        styled_table = styled_table.map(status_color_map, subset=["Status"])

        # Display the styled table
        st.write(styled_table.to_html(), unsafe_allow_html=True)
    else:
        st.write(":red[Data not saved]")

with col2:
    st.button(label="Reset Data", on_click=reset_data)
