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
from utils.velocity_test import (despike, flatline, magnetic_declination,
                                 velocity_cutoff)

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
st.header("Magnetic Declination", divider="blue")
st.write(
    """
The magnetic declination is obtained from World Magnetic Model 2020 (WMM2020).
The python wrapper module `wmm2020` is available from this [Link](https://github.com/space-physics/wmm2020). 
If the magnetic declination is reset, re-run the remaining tests again.
"""
)

if "isMagnetButton" not in st.session_state:
    st.session_state.isMagnetButton = False


def toggle_btns():
    st.session_state.isMagnetButton = not st.session_state.isMagnetButton


with st.form(key="magnet_form"):
    lat = st.number_input("Latitude", -90.0, 90.0, 0.0, step=1.0)
    lon = st.number_input("Longitude", 0.0, 360.0, 0.1, step=1.0, format="%.4f")
    depth = st.number_input("Depth", 0, 1000, 0, step=1)
    year = st.number_input("Year", 1950, 2100, 2024, 1)

    if st.form_submit_button(
        "Compute", on_click=toggle_btns, disabled=st.session_state.isMagnetButton
    ):
        st.session_state.dummyvelocity, mag = magnetic_declination(
            velocity, lat, lon, depth, year
        )
        st.session_state.lat = lat
        st.session_state.lon = lon
        st.session_state.magnetic_dec_depth = depth
        st.session_state.year = year
        st.session_state.angle = np.trunc(mag[0][0])
        st.session_state.isMagnet = True

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
    st.write("Cutoff Applied")
    a = {
        "Max. Zonal Velocity": maxuvel,
        "Max. Meridional Velocity": maxvvel,
        "Max. Vertical Velocity": maxwvel,
    }
    st.write(a)


############## DESPIKE DATA #################
st.header("Despike Data", divider="blue")
despike_kernal = st.number_input(
    "Enter Despike Kernal Size for Median Filter", 0, 1000, 5, 1
)
despike_cutoff = st.number_input("Enter Despike Cutoff (mm/s)", 0, 1000, 150, 1)
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
    st.write("Data Despiked")
    b = {
        "Kernal Size": despike_kernal,
        "Despike Cutoff": despike_cutoff,
    }
    st.write(b)

st.header("Remove Flatline", divider="blue")
flatline_kernal = st.number_input("Enter Flatline Kernal Size", 0, 100, 13, 1)
flatline_cutoff = st.number_input("Enter Flatline deviation", 0, 100, 1, 1)

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


col1, col2 = st.columns([1, 1])
with col1:
    save_button = st.button(label="Save Data")
    if save_button:
        st.session_state.veltest_velocity = np.copy(st.session_state.dummyvelocity)
        st.session_state.velocity_mask = np.copy(st.session_state.maskd)
        st.session_state.isVelocityMask = True
        st.write(":green[Mask data saved]")
    else:
        st.write(":red[Data not saved]")

with col2:
    st.button(label="Reset Data", on_click=reset_data)
