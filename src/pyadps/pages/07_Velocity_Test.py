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
from utils.velocity_test import (
    despike,
    flatline,  # magnetic_declination,
    magdec,
    velocity_cutoff,
    wmm2020api,
    velocity_modifier,
)

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()

flobj = st.session_state.flead
vlobj = st.session_state.vlead
fdata = flobj.fleader
vdata = vlobj.vleader


def reset_velocitytest():
    # Reset Global Test
    st.session_state.isVelocityTest = False

    # Reset Local Tests
    st.session_state.isMagnetCheck_VT = False
    st.session_state.isDespikeCheck_VT = False
    st.session_state.isFlatlineCheck = False
    st.session_state.isCutoffCheck_VT = False

    st.session_state.isVelocityModifiedMagnet = False

    # Page return
    st.session_state.isProfilePageReturn = False
    if not st.session_state.isProfileTest:
        st.session_state.isQCPageReturn = False
    if not st.session_state.isQCTest:
        st.session_state.isSensorPageReturn = False

    # Data Reset
    if st.session_state.isRegridCheck_PT:
        st.session_state.velocity_magnet = st.session_state.velocity_regrid
    elif st.session_state.isVelocityModifiedSound_ST:
        st.session_state.velocity_magnet = st.session_state.velocity_sensor
    else:
        st.session_state.velocity_magnet = st.session_state.velocity

    # Reset Mask
    if st.session_state.isProfileTest:
        if st.session_state.isRegridCheck_PT:
            st.session_state.velocity_mask_default = np.copy(
                st.session_state.profile_mask_regrid
            )
        else:
            st.session_state.velocity_mask_default = np.copy(
                st.session_state.profile_mask
            )
    elif st.session_state.isQCTest:
        st.session_state.velocity_mask_default = np.copy(st.session_state.qc_mask)
    elif st.session_state.isSensorTest:
        st.session_state.velocity_mask_default = np.copy(st.session_state.sensor_mask)
    else:
        st.session_state.velocity_mask_default = np.copy(st.session_state.orig_mask)

    mask = st.session_state.velocity_mask_default
    st.session_state.velocity_mask_temp = np.copy(mask)
    st.session_state.velocity_mask = np.copy(mask)
    st.session_state.velocity_mask_cutoff = np.copy(mask)
    st.session_state.velocity_mask_spike = np.copy(mask)
    st.session_state.velocity_mask_flatline = np.copy(mask)


def hard_reset(option):
    # Reset Global Test
    st.session_state.isVelocityTest = False

    # Reset Local Tests
    st.session_state.isMagnetCheck_VT = False
    st.session_state.isDespikeCheck_VT = False
    st.session_state.isFlatlineCheck = False
    st.session_state.isCutoffCheck_VT = False

    # Page return
    st.session_state.isProfilePageReturn = False
    if not st.session_state.isProfileTest:
        st.session_state.isQCPageReturn = False
    if not st.session_state.isQCTest:
        st.session_state.isSensorPageReturn = False

    # Velocity data reset
    st.session_state.velocity_magnet = st.session_state.velocity

    if option == "Sensor Test":
        st.session_state.velocity_mask_default = np.copy(st.session_state.sensor_mask)
        if st.session_state.isVelocityModifiedSound_ST:
            st.session_state.velocity_magnet = st.session_state.velocity_sensor
    elif option == "QC Test":
        st.session_state.velocity_mask_default = np.copy(st.session_state.qc_mask)
    elif option == "Profile Test":
        st.session_state.velocity_mask_default = np.copy(st.session_state.profile_mask)
        if st.session_state.isRegridCheck_PT:
            st.session_state.velocity_magnet = st.session_state.velocity_regrid
    else:
        st.session_state.velocity_mask_default = np.copy(st.session_state.orig_mask)

    st.session_state.velocity_mask = np.copy(st.session_state.velocity_mask_default)
    st.session_state.velocity_mask_temp = np.copy(
        st.session_state.velocity_mask_default
    )
    st.session_state.velocity_mask_cutoff = np.copy(
        st.session_state.velocity_mask_default
    )
    st.session_state.velocity_mask_spike = np.copy(
        st.session_state.velocity_mask_default
    )
    st.session_state.velocity_mask_flatline = np.copy(
        st.session_state.velocity_mask_default
    )


####### Initialize Mask File ##############
if (
    st.session_state.isProfileTest
    or st.session_state.isQCTest
    or st.session_state.isSensorTest
):
    st.write(":grey[Working on a saved mask file ...]")
    if st.session_state.isVelocityPageReturn:
        st.write(
            ":orange[Warning: Velocity test already completed. Reset to change settings.]"
        )
        reset_selectbox = st.selectbox(
            "Choose reset option",
            ("Profile Test", "QC Test", "Sensor Test", "Default"),
            index=None,
            placeholder="Reset mask to ...",
        )
        if reset_selectbox == "Default":
            st.write("Default mask file selected")
        elif reset_selectbox == "Sensor Test":
            st.write("Sensor Test mask file selected")
        elif reset_selectbox == "QC Test":
            st.write("QC Test mask file selected")
        elif reset_selectbox == "Profile Test":
            st.write("Profile Test mask file selected")

        if reset_selectbox is not None:
            hard_reset(reset_selectbox)

    elif st.session_state.isFirstVelocityVisit:
        reset_velocitytest()
        st.session_state.isFirstVelocityVisit = False
else:
    if st.session_state.isFirstVelocityVisit:
        reset_velocitytest()
        st.session_state.isFirstVelocityVisit = False
    st.write(":grey[Creating a new mask file ...]")


# If data are not regrided use the default one
# if st.session_state.isGridSave:
#     st.session_state.velocity_magnet = np.copy(st.session_state.velocity_regrid)
# else:
#     st.session_state.velocity_magnet = np.copy(st.session_state.velocity)

velocity = st.session_state.velocity_magnet

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Magnetic Declination",
        "Velocity Cutoffs",
        "Despike Data",
        "Remove Flatline",
        "Save & Reset Data",
    ]
)

############ Magnetic Declination ##############
#  Commenting the wmm2020 c based model if needed can be implemented.

# * The magnetic declination is obtained from World Magnetic Model 2020 (WMM2020).
# The python wrapper module `wmm2020` is available from this [Link](https://github.com/space-physics/wmm2020).

with tab1:
    st.header("Magnetic Declination", divider="blue")
    st.write(
        """
        * The pygeomag method uses a python library [pygeomag](https://github.com/boxpet/pygeomag.git) for calculating the magnetic declination.
            * It can work from 2010 till date.
        * The API method utilizes the online magnetic declination service provided by the National Geophysical Data Center (NGDC)
        of the National Oceanic and Atmospheric Administration (NOAA) to calculate the magnetic declination. The service is available at this [link](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#declination).
        Internet connection is necessary for this method to work.
        * According to the year, different models are used for calculating magnetic declination.
            * From 2025 till date, WMM2025 (World Magnetic Model) 
            * From 2019 to 2024 IGRF (International Geomagnetic Reference Field)
            * From 2000 to 2018 EMM (Enhanced Magnetic Model)
            * Before 1999 IGRF
        * In the manual method, the user can directly enter the magnetic declination.
        If the magnetic declination is reset, re-run the remaining tests again.
        """
    )

    # Selecting the method to calculate magnetic declination.
    # method = st.radio("Select a method", ("WMM2020", "API", "Manual"), horizontal=True)
    method = st.radio("Select a method", ("pygeomag", "API", "Manual"), horizontal=True)
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
        if st.session_state.method == "pygeomag":
            # st.session_state.isMagnet = False
            lat = st.number_input("Latitude", -90.0, 90.0, 0.0, step=1.0)
            lon = st.number_input("Longitude", 0.0, 360.0, 0.1, step=1.0, format="%.4f")
            depth = st.number_input("Depth", 0, 1000, 0, step=1)
            year = st.number_input("Year", 2010, 2030, 2025, 1)

        elif st.session_state.method == "API":
            # st.session_state.isMagnet = False
            lat = st.number_input("Latitude", -90.0, 90.0, 0.0, step=1.0)
            lon = st.number_input("Longitude", 0.0, 360.0, 0.1, step=1.0, format="%.4f")
            year = st.number_input("Year", 1950, 2030, 2025, 1)
        else:
            # st.session_state.isMagnet = False
            mag = [[st.number_input("Declination", -180.0, 180.0, 0.0, 0.1)]]
            st.session_state.magnet_user_input_VT = mag

        if st.session_state.method == "Manual":
            button_name = "Accept"
        else:
            button_name = "Compute"

        if st.form_submit_button(
            button_name, on_click=toggle_btns, disabled=st.session_state.isMagnetButton
        ):
            if st.session_state.method == "pygeomag":
                mag = magdec(lat, lon, depth, year)
                st.session_state.velocity_magnet = velocity_modifier(velocity, mag)
                st.session_state.magnet_lat_VT = lat
                st.session_state.magnet_lon_VT = lon
                st.session_state.magnet_year_VT = year
                st.session_state.magnet_depth_VT = depth
                st.session_state.angle = np.round(mag[0][0], decimals=3)
                st.session_state.isMagnetCheck_VT = True
                st.session_state.isButtonClicked = True

            if st.session_state.method == "API":
                try:
                    mag = wmm2020api(lat, lon, year)
                    st.session_state.velocity_magnet = velocity_modifier(velocity, mag)
                    st.session_state.magnet_lat_VT = lat
                    st.session_state.magnet_lon_VT = lon
                    st.session_state.magnet_year_VT = year
                    # st.session_state.angle = np.trunc(mag[0][0])
                    st.session_state.angle = np.round(mag[0][0], decimals=3)
                    st.session_state.isMagnetCheck_VT = True
                    st.session_state.isButtonClicked = True
                except:
                    st.write(
                        ":red[Connection error! please check the internet or use manual method]"
                    )
            else:
                st.session_state.velocity_magnet = velocity_modifier(velocity, mag)
                st.session_state.angle = np.round(mag[0][0], decimals=3)
                st.session_state.isMagnetCheck_VT = True
                st.session_state.isButtonClicked = True

        if st.session_state.isMagnetCheck_VT:
            st.write(f"Magnetic declination: {st.session_state.angle}\u00b0")
            st.write(":green[Magnetic declination correction applied to velocities]")

    magnet_button_reset = st.button(
        "Reset Magnetic Declination",
        on_click=toggle_btns,
        disabled=not st.session_state.isMagnetButton,
    )
    if magnet_button_reset:
        st.session_state.velocity_magnet = np.copy(velocity)
        st.session_state.isMagnetCheck_VT = False
        st.session_state.isButtonClicked = False

with tab2:
    ############# Velocity Cutoffs #################
    st.header("Velocity Cutoffs", divider="blue")
    st.write(
        """
    Drop velocities whose magnitude is larger than the threshold.
    """
    )
    with st.form(key="cutbin_form"):
        maxuvel = st.number_input(
            "Maximum Zonal Velocity Cutoff (cm/s)", 0, 2000, 250, 1
        )
        maxvvel = st.number_input(
            "Maximum Meridional Velocity Cutoff (cm/s)", 0, 2000, 250, 1
        )
        maxwvel = st.number_input(
            "Maximum Vertical Velocity Cutoff (cm/s)", 0, 2000, 15, 1
        )
        submit_cutoff = st.form_submit_button(label="Submit")

    if submit_cutoff:
        velocity = st.session_state.velocity_magnet
        st.session_state.maxuvel_VT = maxuvel
        st.session_state.maxvvel_VT = maxvvel
        st.session_state.maxwvel_VT = maxwvel

        st.session_state.velocity_mask_cutoff = velocity_cutoff(
            velocity[0, :, :], st.session_state.velocity_mask_temp, cutoff=maxuvel
        )
        st.session_state.velocity_mask_cutoff = velocity_cutoff(
            velocity[1, :, :], st.session_state.velocity_mask_temp, cutoff=maxvvel
        )
        st.session_state.velocity_mask_cutoff = velocity_cutoff(
            velocity[2, :, :], st.session_state.velocity_mask_temp, cutoff=maxwvel
        )
        st.session_state.velocity_mask_temp = np.copy(
            st.session_state.velocity_mask_cutoff
        )
        st.session_state.isCutoffCheck_VT = True

    if st.session_state.isCutoffCheck_VT:
        st.success("Cutoff Applied")
        a = {
            "Max. Zonal Velocity": maxuvel,
            "Max. Meridional Velocity": maxvvel,
            "Max. Vertical Velocity": maxwvel,
        }
        st.write(a)

    def reset_button_cutoff():
        st.session_state.isCutoffCheck_VT = False
        st.session_state.velocity_mask_temp = np.copy(
            st.session_state.velocity_mask_default
        )
        st.session_state.velocity_mask_cutoff = np.copy(
            st.session_state.velocity_mask_default
        )

    reset_cutoff = st.button("Reset Cutoff to default", on_click=reset_button_cutoff)
    if reset_cutoff:
        st.info("Cutoff Test is reset")


with tab3:
    ############## DESPIKE DATA #################
    st.header("Despike Data", divider="blue")
    st.write("""A rolling median filter is applied to remove spikes from the data.
    The kernal size determines the number of ensembles (time interval) for the filter window.
    The standard deviation specifies the maximum allowable deviation to remove the spike.""")

    # time_interval = pd.Timedelta(st.session_state.date[-1] - st.session_state.date[0]).seconds/(3600*st.session_state.head.ensembles)

    st.write("Time interval: ", st.session_state.date[1] - st.session_state.date[0])

    despike_kernal = st.number_input(
        "Enter Despike Kernal Size for Median Filter",
        0,
        st.session_state.head.ensembles,
        5,
        1,
    )

    despike_cutoff = st.number_input(
        "Standard Deviation Cutoff for Spike Removal", 0.1, 10.0, 3.0, 0.1
    )
    despike_button = st.button("Despike")
    if despike_button:
        st.session_state.despike_kernal_VT = despike_kernal
        st.session_state.despike_cutoff_VT = despike_cutoff

        st.session_state.velocity_mask_despike = despike(
            velocity[0, :, :],
            st.session_state.velocity_mask_temp,
            kernal_size=despike_kernal,
            cutoff=despike_cutoff,
        )
        st.session_state.velocity_mask_despike = despike(
            velocity[1, :, :],
            st.session_state.velocity_mask_temp,
            kernal_size=despike_kernal,
            cutoff=despike_cutoff,
        )

        # Reset the temporary mask
        st.session_state.velocity_mask_temp = np.copy(
            st.session_state.velocity_mask_despike
        )
        st.session_state.isDespikeCheck_VT = True

    if st.session_state.isDespikeCheck_VT:
        st.success("Data Despiked")
        b = {
            "Kernal Size": despike_kernal,
            "Despike Cutoff": despike_cutoff,
        }
        st.write(b)

    def reset_button_despike():
        st.session_state.isDespikeCheck_VT = False
        if st.session_state.isCutoffCheck_VT:
            st.session_state.velocity_mask_temp = np.copy(
                st.session_state.velocity_mask_cutoff
            )
            st.session_state.velocity_mask_despike = np.copy(
                st.session_state.velocity_mask_cutoff
            )
        else:
            st.session_state.velocity_mask_temp = np.copy(
                st.session_state.velocity_mask_default
            )
            st.session_state.velocity_mask_despike = np.copy(
                st.session_state.velocity_mask_default
            )

    reset_despike = st.button("Reset Despike to default", on_click=reset_button_despike)
    if reset_despike:
        st.info("Despike Test is reset")

with tab4:
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
        st.session_state.flatline_kernal_VT = flatline_kernal
        st.session_state.flatline_cutoff_VT = flatline_cutoff

        st.session_state.velocity_mask_flatline = flatline(
            velocity[0, :, :],
            st.session_state.velocity_mask_temp,
            kernal_size=flatline_kernal,
            cutoff=flatline_cutoff,
        )
        st.session_state.velocity_mask_flatline = flatline(
            velocity[1, :, :],
            st.session_state.velocity_mask_temp,
            kernal_size=flatline_kernal,
            cutoff=flatline_cutoff,
        )
        st.session_state.velocity_mask_flatline = flatline(
            velocity[2, :, :],
            st.session_state.velocity_mask_temp,
            kernal_size=flatline_kernal,
            cutoff=flatline_cutoff,
        )
        # Modify the temporary mask file
        st.session_state.velocity_mask_temp = np.copy(
            st.session_state.velocity_mask_flatline
        )
        st.session_state.isFlatlineCheck = True

    if st.session_state.isFlatlineCheck:
        st.success("Flatline Removed")
        b = {
            "Kernal Size": flatline_kernal,
            "Flatline Cutoff": flatline_cutoff,
        }
        st.write(b)

    def reset_button_flatline():
        st.session_state.isFlatlineCheck = False
        if st.sesion_state.isDespikeCheck_VT:
            st.session_state.velocity_mask_temp = np.copy(
                st.session_state.velocity_mask_despike
            )
            st.session_state.velocity_mask_flatline = np.copy(
                st.session_state.velocity_mask_despike
            )
        elif st.session_state.isCutoffCheck_VT:
            st.session_state.velocity_mask_temp = np.copy(
                st.session_state.velocity_mask_cutoff
            )
            st.session_state.velocity_mask_flatline = np.copy(
                st.session_state.velocity_mask_cutoff
            )
        else:
            st.session_state.velocity_mask_temp = np.copy(
                st.session_state.velocity_mask_default
            )
            st.session_state.velocity_mask_flatline = np.copy(
                st.session_state.velocity_mask_default
            )

    reset_despike = st.button(
        "Reset Flatline to default", on_click=reset_button_flatline
    )
    if reset_despike:
        st.info("Flatline Test is reset")


##################### SAVE DATA ###################
with tab5:
    st.header("Save & Reset Data", divider="blue")

    def save_velocitytest():
        st.session_state.isVelocityTest = True
        st.session_state.isFirstVelocityVisit = False
        st.session_state.velocity_mask = st.session_state.velocity_mask_temp

        st.session_state.isSensorPageReturn = True
        st.session_state.isQCPageReturn = True
        st.session_state.isProfilePageReturn = True

    col1, col2 = st.columns([1, 1])
    with col1:
        save_button = st.button(label="Save Data", on_click=save_velocitytest)
        if save_button:
            st.write(":green[Mask data saved]")
            # Status Summary Table
            status_summary = pd.DataFrame(
                [
                    [
                        "Magnetic Declination",
                        "True" if st.session_state.isButtonClicked else "False",
                    ],
                    [
                        "Velocity Cutoffs",
                        "True" if st.session_state.isCutoffCheck_VT else "False",
                    ],
                    [
                        "Despike Data",
                        "True" if st.session_state.isDespikeCheck_VT else "False",
                    ],
                    [
                        "Remove Flatline",
                        "True" if st.session_state.isFlatlineCheck else "False",
                    ],
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
            styled_table = status_summary.style.set_properties(
                **{"text-align": "center"}
            )
            styled_table = styled_table.map(status_color_map, subset=["Status"])

            # Display the styled table
            st.write(styled_table.to_html(), unsafe_allow_html=True)
        else:
            st.write(":red[Data not saved]")

    with col2:
        st.button(label="Reset Data", on_click=reset_velocitytest)
        st.info("Velocity test reset to default")
