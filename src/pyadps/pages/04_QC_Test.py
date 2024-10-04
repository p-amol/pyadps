import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from utils.signal_quality import ev_check, false_target, pg_check, qc_check

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()

# `mask` holds the temporary changes in the page
# `qcmask` holds the final changes in the page
if "mask" not in st.session_state:
    st.session_state.mask = np.copy(st.session_state.orig_mask)

if not st.session_state.isQCMask:
    st.write(":grey[Creating a new mask file ...]")
    st.session_state.qc_mask = np.copy(st.session_state.orig_mask)
    st.session_state.isSubmit = False
else:
    st.write(":grey[Working on a saved mask file ...]")
    st.write(":orange[WARNING! QC test already completed. Reset to change settings.]")
    reset_button1 = st.button("Reset Mask Data")
    if reset_button1:
        st.session_state.mask = np.copy(st.session_state.orig_mask)
        st.session_state.qc_mask = np.copy(st.session_state.orig_mask)
        st.write(":green[Mask data is reset to default]")

if "isThresh" not in st.session_state:
    st.session_state.isThresh = False

# Load data
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
    
#########  NOISE FLOOR IDENTIFICATION ##############
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

st.header("Quality Control Tests", divider="blue")

left, right = st.columns([1, 1])
with left:
    st.write("Thresholds used during deployment")
    fdata = st.session_state.flead.field()
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

        ct = st.number_input(
            "Select Correlation Threshold",
            0,
            255,
            fdata["Correlation Thresh"],
        )

        evt = st.number_input(
            "Select Error Velocity Threshold",
            0,
            9999,
            fdata["Error Velocity Thresh"],
        )

        et = st.number_input(
            "Select Echo Intensity Threshold",
            0,
            255,
            0,
        )

        ft = st.number_input(
            "Select False Target Threshold",
            0,
            255,
            fdata["False Target Thresh"],
        )

        option = st.selectbox(
            "Would you like to use a three-beam solution?", (True, False)
        )

        pgt = st.number_input(
            "Select Percent Good Threshold",
            0,
            100,
            fdata["Percent Good Min"],
        )
        submit_button = st.form_submit_button(label="Submit")


mask = st.session_state.mask
with left:
    if submit_button:
        st.session_state.newthresh = pd.DataFrame(
            [
                ["Correlation", str(ct)],
                ["Error Velocity", str(evt)],
                ["Echo Intensity", str(et)],
                ["False Target", str(ft)],
                ["Three Beam", str(option)],
                ["Percentage Good", str(pgt)],
            ],
            columns=["Threshold", "Values"],
        )
        st.session_state.isThresh = True
        # st.write(st.session_state.newthresh)

        mask = pg_check(pgood, mask, pgt, threebeam=option)
        mask = qc_check(correlation, mask, ct)
        mask = qc_check(echo, mask, et)
        mask = ev_check(velocity[3, :, :], mask, evt)
        mask = false_target(echo, mask, ft, threebeam=True)
        st.session_state.mask = mask

    if st.session_state.isThresh:
        st.write("Current Thresholds")
        st.write(st.session_state.newthresh)



st.header("Mask File", divider="blue")
st.write(
"""
Displayed the mask file. 
Ensure to save any necessary changes or apply additional thresholds if needed.
"""
)


if st.button("Display mask file"):
    st.header("Default Mask File", divider="blue")
    st.write(
    """
ADCP assigns missing values based on thresholds set before deployment.
These values cannot be recovered and the default 
"""
)
    fillplot_plotly(st.session_state.orig_mask, colorscale="greys")


    st.header("Update Mask File", divider="blue")
    st.write(
    """
Update, display and save the updated mask file after applying threshold.
If thresholds are not saved, default mask file is used. 
"""
)
# values, counts = np.unique(mask, return_counts=True)
    fillplot_plotly(st.session_state.mask, colorscale="greys")

col1, col2 = st.columns([1, 1])

with col1:
    save_mask_button = st.button(label="Save Mask Data")

    if save_mask_button:
        # st.session_state.mask = mask
        st.session_state.qc_mask = np.copy(st.session_state.mask)
        st.session_state.isQCMask = True
        st.session_state.isProfileMask = False
        st.session_state.isGridSave = False
        st.session_state.isVelocityMask = False
        st.write(":green[Mask file saved]")
    else:
        st.write(":red[Mask data not saved]")
with col2:
    reset_mask_button = st.button("Reset mask Data")
    if reset_mask_button:
        st.session_state.mask = np.copy(st.session_state.orig_mask)
        st.session_state.isQCMask = False
        st.session_state.isGrid = False
        st.session_state.isProfileMask = False
        st.session_state.isVelocityMask = False
        st.write(":green[Mask data is reset to default]")
