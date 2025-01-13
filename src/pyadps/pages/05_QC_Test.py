import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from streamlit.runtime.state import session_state
from utils.signal_quality import (
    ev_check,
    false_target,
    pg_check,
    echo_check,
    correlation_check,
)

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()


def reset_qctest():
    # Reset Global Test
    st.session_state.isQCTest = False
    # Reset Local Tests
    st.session_state.isQCCheck_QCT = False

    # Reset Data
    st.session_state.isBeamModified_QCT = False
    st.session_state.beam_direction_QCT = st.session_state.beam_direction

    # As QC test is not saved the Sensor Page Returns is set to False
    st.session_state.isSensorPageReturn = False

    # Reset Mask Data
    # Copy the sensor mask if sensor mask completed.
    # Else copy the default mask.
    if st.session_state.isSensorTest:
        st.session_state.qc_mask = np.copy(st.session_state.sensor_mask)
        st.session_state.qc_mask_temp = np.copy(st.session_state.sensor_mask)
    else:
        st.session_state.qc_mask = np.copy(st.session_state.orig_mask)
        st.session_state.qc_mask_temp = np.copy(st.session_state.orig_mask)


def hard_reset(option):
    # Reset Global Test
    st.session_state.isQCTest = False
    # Reset Local Tests
    st.session_state.isQCCheck_QCT = False
    st.session_state.isBeamModified_QCT = False
    # Reset Data
    st.session_state.beam_direction_QCT = st.session_state.beam_direction

    st.session_state.isSensorPageReturn = False

    # Reset Mask Data based on user options
    if option == "Sensor Test":
        st.session_state.qc_mask = np.copy(st.session_state.sensor_mask)
        st.session_state.qc_mask_temp = np.copy(st.session_state.sensor_mask)

    elif option == "Default":
        st.session_state.qc_mask = np.copy(st.session_state.orig_mask)
        st.session_state.qc_mask_temp = np.copy(st.session_state.orig_mask)


def save_qctest():
    st.session_state.qc_mask = np.copy(st.session_state.qc_mask_temp)
    st.session_state.isQCTest = True
    st.session_state.isProfileMask = False
    st.session_state.isGridSave = False
    st.session_state.isVelocityMask = False
    # Indicate previous pages that Test has been carried out
    st.session_state.isSensorPageReturn = True


def qc_submit():
    # st.write(st.session_state.newthresh)
    st.session_state.isQCCheck_QCT = True

    # First Quality check of the page
    mask = np.copy(st.session_state.qc_mask_temp)
    # if st.session_state.isSensorTest:
    #     mask = np.copy(st.session_state.sensor_mask)
    # else:
    #     mask = np.copy(st.session_state.default_mask)

    ds = st.session_state.ds
    pgt = st.session_state.pgt_QCT
    ct = st.session_state.ct_QCT
    et = st.session_state.et_QCT
    evt = st.session_state.evt_QCT
    ft = st.session_state.ft_QCT
    is3beam = st.session_state.is3beam_QCT
    mask = pg_check(ds, mask, pgt, threebeam=is3beam)
    mask = correlation_check(ds, mask, ct)
    mask = echo_check(ds, mask, et)
    mask = ev_check(ds, mask, evt)
    mask = false_target(ds, mask, ft, threebeam=True)
    # Store the processed mask in a temporary mask
    st.session_state.qc_mask_temp = mask


if st.session_state.isSensorTest:
    st.write(":grey[Working on a saved mask file ...]")
    if st.session_state.isQCPageReturn:
        st.write(
            ":orange[Warning: QC test already completed. Reset the mask file to change settings.]"
        )
        reset_selectbox = st.selectbox(
            "Choose reset option",
            ("Sensor Test", "Default"),
            index=None,
            placeholder="Reset mask to ...",
        )
        if reset_selectbox is not None:
            hard_reset(reset_selectbox)
    elif st.session_state.isFirstQCVisit:
        reset_qctest()
        st.session_state.isFirstQCVisit = False
else:
    if st.session_state.isFirstQCVisit:
        # This will rest to the default mask file
        reset_qctest()
        st.session_state.isFirstQCVisit = False
    st.write(":orange[Creating a new mask file ...]")


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


@st.cache_data
def fillplot_plotly(data, colorscale="balance"):
    fig = FigureResampler(go.Figure())
    data1 = np.where(data == -32768, np.nan, data)
    fig.add_trace(
        go.Heatmap(z=data1[:, 0:-1], x=x, y=y, colorscale=colorscale, hoverongaps=False)
    )
    st.plotly_chart(fig)


@st.cache_data
def lineplot(data, title, slope=None, xaxis="time"):
    if xaxis == "time":
        xdata = st.session_state.date
    else:
        xdata = st.session_state.ensemble_axis
    scatter_trace = go.Scatter(
        x=xdata,
        y=data,
        mode="markers",
        name=title,
        marker=dict(color="blue", size=10),  # Customize marker color and size
    )
    # Create the slope line trace
    if slope is not None:
        line_trace = go.Scatter(
            x=xdata,
            y=slope,
            mode="lines",
            name="Slope Line",
            line=dict(color="red", width=2, dash="dash"),  # Dashed red line
        )
        fig = go.Figure(data=[scatter_trace, line_trace])
    else:
        fig = go.Figure(data=[scatter_trace])

    st.plotly_chart(fig)


@st.cache_data
def plot_noise(dep=0, rec=-1):
    n = dep
    m = rec
    colorleft = [
        "rgb(240, 255, 255)",
        "rgb(115, 147, 179)",
        "rgb(100, 149, 237)",
        "rgb(15, 82, 186)",
    ]
    colorright = [
        "rgb(250, 200, 152)",
        "rgb(255, 165, 0)",
        "rgb(255, 95, 31)",
        "rgb(139, 64, 0)",
    ]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Deployment Ensemble ({x[n]+1})",
            f"Recovery Ensemble ({x[m]+1})",
        ],
    )
    for i in range(4):
        fig.add_trace(
            go.Scatter(
                x=echo[i, :, n],
                y=y,
                name=f"Beam (D) {i+1}",
                line=dict(color=colorleft[i]),
            ),
            row=1,
            col=1,
        )
    for i in range(4):
        fig.add_trace(
            go.Scatter(
                x=echo[i, :, m],
                y=y,
                name=f"Beam (R)  {i+1}",
                line=dict(color=colorright[i]),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(height=600, width=800, title_text="Echo Intensity")
    fig.update_xaxes(title="Echo (count)")
    fig.update_yaxes(title="Cells")
    st.plotly_chart(fig)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Noise Floor Identification",
        "QC Tests",
        "Display Mask",
        "Fix Orientation",
        "Save Data",
    ]
)
#########  NOISE FLOOR IDENTIFICATION ##############
with tab1:
    dn = rn = 1
    st.header("Noise Floor Identification", divider="blue")
    st.write(
        """
        If the ADCP has collected data from the air either 
        before deployment or after recovery, this data can 
        be used to estimate the echo intensity threshold. 
        The plots below show the echo intensity from the first 
        and last ensembles. The noise level is typically around 
        30-40 counts throughout the entire profile.
    """
    )
    dn = st.number_input("Deployment Ensemble", x[0] + 1, x[-1] + 1, x[0] + 1)
    # r = st.number_input("Recovery Ensemble", -1 * (x[-1] + 1), -1 * (x[0] + 1), -1)
    rn = st.number_input("Recovery Ensemble", x[0] + 1, x[-1] + 1, x[-1] + 1)
    dn = dn - 1
    rn = rn - 1

    plot_noise(dep=dn, rec=rn)


################## QC Test ###################
with tab2:
    st.header("Quality Control Tests", divider="blue")
    st.write("")

    left, right = st.columns([1, 1])
    with left:
        st.write(""" Teledyne RDI recommends these quality control tests, 
                    some of which can be configured before deployment. 
                    The pre-deployment values configured for the ADCP are listed 
                    in the table below. The noise-floor identification graph above 
                    can assist in determining the echo intensity threshold. 
                    For more information about these tests, 
                    refer to *Acoustic Doppler Current Profiler Principles of 
                    Operation: A Practical Primer* by Teledyne RDI.""")
        fdata = st.session_state.flead.field()
        st.divider()
        st.write(":blue-background[Additional Information:]")
        st.write(f"Number of Pings per Ensemble: `{fdata["Pings"]}`")
        st.write(f"Number of Beams: `{fdata["Beams"]}`")
        st.divider()
        st.write(":red-background[Thresholds used during deployment:]")
        thresh = pd.DataFrame(
            [
                ["Correlation", fdata["Correlation Thresh"]],
                ["Error Velocity", fdata["Error Velocity Thresh"]],
                ["Echo Intensity", 0],
                ["False Target", fdata["False Target Thresh"]],
                ["Percentage Good", fdata["Percent Good Min"]],
            ],
            columns=["Threshold", "Values"],
        )

        st.write(thresh)

    with right:
        with st.form(key="my_form"):
            st.write("Would you like to apply new threshold?")

            st.session_state.ct_QCT = st.number_input(
                "Select Correlation Threshold",
                0,
                255,
                fdata["Correlation Thresh"],
            )

            st.session_state.evt_QCT = st.number_input(
                "Select Error Velocity Threshold",
                0,
                9999,
                fdata["Error Velocity Thresh"],
            )

            st.session_state.et_QCT = st.number_input(
                "Select Echo Intensity Threshold",
                0,
                255,
                0,
            )

            st.session_state.ft_QCT = st.number_input(
                "Select False Target Threshold",
                0,
                255,
                fdata["False Target Thresh"],
            )

            st.session_state.is3beam_QCT = st.selectbox(
                "Would you like to use a three-beam solution?", (True, False)
            )

            st.session_state.pgt_QCT = st.number_input(
                "Select Percent Good Threshold",
                0,
                100,
                fdata["Percent Good Min"],
            )
            submit_button = st.form_submit_button(label="Submit", on_click=qc_submit)

    # mask = st.session_state.qc_mask_temp
    with left:
        if submit_button:
            st.session_state.isQCCheck_QCT = True
            st.session_state.newthresh = pd.DataFrame(
                [
                    ["Correlation", str(st.session_state.ct_QCT)],
                    ["Error Velocity", str(st.session_state.evt_QCT)],
                    ["Echo Intensity", str(st.session_state.et_QCT)],
                    ["False Target", str(st.session_state.ft_QCT)],
                    ["Three Beam", str(st.session_state.is3beam_QCT)],
                    ["Percentage Good", str(st.session_state.pgt_QCT)],
                ],
                columns=["Threshold", "Values"],
            )

        if st.session_state.isQCCheck_QCT:
            st.write(":green-background[Current Thresholds]")
            st.write(st.session_state.newthresh)


with tab3:
    st.header("Mask File", divider="blue")
    st.write(
        """
    Display the mask file. 
    Ensure to save any necessary changes or apply additional thresholds if needed.
    """
    )

    leftplot, rightplot = st.columns([1, 1])
    if st.button("Display mask file"):
        with leftplot:
            st.subheader("Default Mask File")
            st.write(
                """
                    CAPTION:
                    ADCP assigns missing values based on thresholds
                    set before deployment. These values cannot be
                    recovered and are part of default mask file.
                """
            )
            fillplot_plotly(st.session_state.orig_mask, colorscale="greys")
        with rightplot:
            st.subheader("Updated Mask File")
            # values, counts = np.unique(mask, return_counts=True)
            st.write(
                """
                    CAPTION:
                    Updated mask displayed after applying threshold.
                    If thresholds are not saved, default mask
                    is displayed. 
                """
            )
            fillplot_plotly(st.session_state.qc_mask_temp, colorscale="greys")

with tab4:
    ################## Fix Orientation ###################
    st.subheader("Fix Orientation", divider="orange")

    if st.session_state.beam_direction == "Up":
        beamalt = "Down"
    else:
        beamalt = "Up"
    st.write(
        f"The current orientation of ADCP is `{st.session_state.beam_direction}`. Use the below option to correct the orientation."
    )

    beamdir_select = st.radio(f"Change orientation to {beamalt}", ["No", "Yes"])
    if beamdir_select == "Yes":
        st.session_state.beam_direction_QCT = beamalt
        st.session_state.isBeamModified_QCT = True
        st.write(f"The orientation changed to `{st.session_state.beam_direction_QCT}`")

with tab5:
    ################## Save Button #############
    st.header("Save Data", divider="blue")
    col1, col2 = st.columns([1, 1])
    with col1:
        save_mask_button = st.button(label="Save Mask Data", on_click=save_qctest)

        if save_mask_button:
            # st.session_state.qc_mask_temp = mask
            st.success("Mask file saved")
            # Table summarizing changes
            changes_summary = pd.DataFrame(
                [
                    [
                        "Quality Control Tests",
                        "True" if st.session_state.isQCCheck_QCT else "False",
                    ],
                    ["Fix Orientation", st.session_state.beam_direction_QCT],
                ],
                columns=["Test", "Status"],
            )

            # Define a mapping function for styling
            def status_color_map(value):
                if value == "True":
                    return "background-color: green; color: white"
                elif value == "False":
                    return "background-color: red; color: white"
                elif value == "Up":
                    return "background-color: blue; color: white"
                elif value == "Down":
                    return "background-color: orange; color: white"
                else:
                    return ""

            # Apply styles using Styler.apply
            styled_table = changes_summary.style.set_properties(
                **{"text-align": "center"}
            )
            styled_table = styled_table.map(status_color_map, subset=["Status"])

            # Display the styled table
            st.write(styled_table.to_html(), unsafe_allow_html=True)
        else:
            st.warning("Mask data not saved")
    with col2:
        reset_mask_button = st.button("Reset mask Data", on_click=reset_qctest)
        if reset_mask_button:
            # st.session_state.qc_mask_temp = np.copy(st.session_state.orig_mask)
            # st.session_state.isQCCheck_QCT = False
            # st.session_state.isQCTest = False
            # st.session_state.isGrid = False
            # st.session_state.isProfileMask = False
            # st.session_state.isVelocityMask = False
            st.success("Mask data is reset to default")
