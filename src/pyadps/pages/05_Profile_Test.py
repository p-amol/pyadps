import numpy as np
# import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from utils.profile_test import side_lobe_beam_angle
from utils.regrid import regrid2d, regrid3d
from utils.signal_quality import default_mask

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()

# `maskp` holds the temporary changes in the page
# `profile_mask`
if "maskp" not in st.session_state:
    if "qc_mask" not in st.session_state:
        st.session_state.maskp = np.copy(st.session_state.orig_mask)
    else:
        st.session_state.maskp = np.copy(st.session_state.qc_mask)


if st.session_state.isQCMask:
    st.write(":grey[Working on a saved mask file ...]")
    if st.session_state.isProfileMask:
        st.write(
            ":orange[Warning: Profile test already completed. Reset to change settings.]"
        )
        reset_selectbox = st.selectbox(
            "Choose reset option",
            ("QC Test", "Default"),
            index=None,
            placeholder="Reset mask to ...",
        )
        if reset_selectbox == "Default":
            st.write("Default mask file selected")
            st.session_state.maskp = st.session_state.orig_mask
        elif reset_selectbox == "QC Test":
            st.write("QC Test mask file selected")
            st.session_state.maskp = st.session_state.qc_mask
        else:
            st.session_state.maskp = st.session_state.profile_mask
    else:
        st.session_state.maskp = st.session_state.qc_mask
else:
    st.write(":orange[Creating a new mask file ...]")

mask = st.session_state.maskp

# Load data
flobj = st.session_state.flead
vlobj = st.session_state.vlead
velocity = st.session_state.velocity
echo = st.session_state.echo
correlation = st.session_state.correlation
pgood = st.session_state.pgood
fdata = flobj.fleader
vdata = vlobj.vleader


ensembles = st.session_state.head.ensembles
cells = flobj.field()["Cells"]
x = np.arange(0, ensembles, 1)
y = np.arange(0, cells, 1)

# Regrided data
if "velocity_regrid" not in st.session_state:
    st.session_state.echo_regrid = np.copy(echo)
    st.session_state.velocity_regrid = np.copy(velocity)
    st.session_state.correlation_regrid = np.copy(correlation)
    st.session_state.pgood_regrid = np.copy(pgood)
    st.session_state.mask_regrid = np.copy(mask)


# @st.cache_data
def fillplot_plotly(
    data, title="data", maskdata=None, missing=-32768, colorscale="balance"
):
    fig = FigureResampler(go.Figure())
    data1 = np.where(data == missing, np.nan, data)
    fig.add_trace(
        go.Heatmap(
            z=data1,
            x=x,
            y=y,
            colorscale=colorscale,
            hoverongaps=False,
        )
    )
    if mask is not None:
        fig.add_trace(
            go.Heatmap(
                z=maskdata,
                x=x,
                y=y,
                colorscale="gray",
                hoverongaps=False,
                showscale=False,
                opacity=0.5,
            )
        )
    fig.update_layout(
        xaxis=dict(showline=True, mirror=True),
        yaxis=dict(showline=True, mirror=True),
        title_text=title,
    )
    fig.update_xaxes(title="Ensembles")
    fig.update_yaxes(title="Depth Cells")
    st.plotly_chart(fig)


def fillselect_plotly(data, title="data", colorscale="balance"):
    fig = FigureResampler(go.Figure())
    data1 = np.where(data == -32768, None, data)
    fig.add_trace(
        go.Heatmap(
            z=data1,
            x=x,
            y=y,
            colorscale=colorscale,
            hoverongaps=False,
        )
    )
    # fig.add_trace(
    #     go.Scatter(x=X, y=Y, marker=dict(color="black", size=16), mode="lines+markers")
    # )
    fig.update_layout(
        xaxis=dict(showline=True, mirror=True),
        yaxis=dict(showline=True, mirror=True),
        title_text=title,
    )
    fig.update_xaxes(title="Ensembles")
    fig.update_yaxes(title="Depth Cells")
    fig.update_layout(clickmode="event+select")
    event = st.plotly_chart(fig, key="1", on_select="rerun", selection_mode="box")

    return event


@st.cache_data
def trim_ends(start_ens=0, end_ens=0, ens_range=20):
    depth = vdata["Depth of Transducer"] / 10
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Deployment Ensemble",
            "Recovery Ensemble",
        ],
    )
    fig.add_trace(
        go.Scatter(
            x=x[0:ens_range],
            y=depth[0:ens_range],
            name="Deployment",
            mode="markers",
            marker=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x[-1 * ens_range :],
            y=depth[-1 * ens_range :],
            name="Recovery",
            mode="markers",
            marker=dict(color="#17becf"),
        ),
        row=1,
        col=2,
    )

    if start_ens > x[0]:
        fig.add_trace(
            go.Scatter(
                x=x[0:start_ens],
                y=depth[0:start_ens],
                name="Selected Points (D)",
                mode="markers",
                marker=dict(color="red"),
            ),
            row=1,
            col=1,
        )

    if end_ens < x[-1] + 1:
        fig.add_trace(
            go.Scatter(
                x=x[end_ens : x[-1] + 1],
                y=depth[end_ens : x[-1] + 1],
                name="Selected Points (R)",
                mode="markers",
                marker=dict(color="orange"),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(height=600, width=800, title_text="Transducer depth")
    fig.update_xaxes(title="Ensembles")
    fig.update_yaxes(title="Depth (m)")
    st.plotly_chart(fig)


st.header("Profile Test")

############## TRIM ENDS #################
st.header("Trim Ends", divider="blue")
n = 20
m = 20
if "update_mask" not in st.session_state:
    st.session_state.update_mask = False
    st.session_state.endpoints = None
if "update_mask_cutbin" not in st.session_state:
    st.session_state.update_mask_cutbin = False

ens_range = st.number_input("Change range", x[0], x[-1], 20)
start_ens = st.slider("Deployment Ensembles", 0, ens_range, 0)
end_ens = st.slider("Recovery Ensembles", x[-1] - ens_range, x[-1] + 1, x[-1] + 1)

n = int(ens_range)

if start_ens or end_ens:
    trim_ends(start_ens=start_ens, end_ens=end_ens, ens_range=n)
    st.session_state.update_mask = False

update_mask = st.button("Update mask data")
if update_mask:
    if start_ens > 0:
        mask[:, :start_ens] = 1

    if end_ens < x[-1]:
        mask[:, end_ens:] = 1

    st.session_state.ens_range = ens_range
    st.session_state.start_ens = start_ens
    st.session_state.end_ens = end_ens
    st.session_state.maskp = mask
    st.write(":green[mask data updated]")
    st.session_state.endpoints = np.array(
        [st.session_state.start_ens, st.session_state.end_ens]
    )
    st.write(st.session_state.endpoints)
    st.session_state.update_mask = True

if not st.session_state.update_mask:
    st.write(":red[mask data not updated]")


############  CUT BINS (SIDE LOBE) ############################
st.header("Cut Bins: Side Lobe Contamination", divider="blue")
st.write(
    """
The side lobe echos from hard surface such as sea surface or bottom of the ocean can contaminate
data closer to this region. The data closer to the surface or bottom can be removed using 
the relation between beam angle and the thickness of the contaminated layer.
"""
)

# Reset mask
mask = st.session_state.maskp
beam = st.radio("Select beam", (1, 2, 3, 4), horizontal=True)
beam = beam - 1
st.session_state.beam = beam
fillplot_plotly(echo[beam, :, :], title="Echo Intensity")

with st.form(key="cutbin_form"):
    extra_cells = st.number_input("Additional Cells to Delete", 0, 10, 0)
    cut_bins_mask = st.form_submit_button(label="Cut bins")

    if cut_bins_mask:
        st.session_state.extra_cells = extra_cells
        mask = side_lobe_beam_angle(flobj, vlobj, mask, extra_cells=extra_cells)
        fillplot_plotly(
            echo[beam, :, :],
            title="Echo Intensity (Masked)",
            maskdata=mask,
        )
        fillplot_plotly(mask, colorscale="greys", title="Mask Data")

update_mask_cutbin = st.button("Update mask file after cutbin")
if update_mask_cutbin:
    st.session_state.maskp = mask
    st.write(":green[mask file updated]")
    st.session_state.update_mask_cutbin = True

if not st.session_state.update_mask_cutbin:
    st.write(":red[mask file not updated]")


############ CUT BINS: Manual #################
st.header("Cut Bins: Manual", divider="blue")


############ REGRID ###########################################
st.header("Regrid Depth Cells", divider="blue")

st.write(
    """
When the ADCP buoy has vertical oscillations (greater than depth cell size), 
the depth bins has to be regridded based on the pressure sensor data. The data
can be regrided either till the surface or till the last bin. 
If the `bin` option is selected, ensure that the end data are trimmed.
"""
)

last_cell = st.radio(
    "Select the depth of last bin for regridding", ("Bin", "Surface"), horizontal=True
)
st.session_state.last_cell = last_cell
st.write(last_cell)
regrid_button = st.button(label="Regrid Data")


if regrid_button:
    st.write(st.session_state.endpoints)
    z, st.session_state.velocity_regrid = regrid3d(
        flobj, vlobj, velocity, -32768, trimends=st.session_state.endpoints
    )
    st.write(":grey[Regrided velocity ...]")
    z, st.session_state.echo_regrid = regrid3d(
        flobj, vlobj, echo, -32768, trimends=st.session_state.endpoints
    )
    st.write(":grey[Regrided echo intensity ...]")
    z, st.session_state.correlation_regrid = regrid3d(
        flobj, vlobj, correlation, -32768, trimends=st.session_state.endpoints
    )
    st.write(":grey[Regrided correlation...]")
    z, st.session_state.pgood_regrid = regrid3d(
        flobj, vlobj, pgood, -32768, trimends=st.session_state.endpoints
    )
    st.write(":grey[Regrided percent good...]")
    z, st.session_state.mask_regrid = regrid2d(
        flobj, vlobj, mask, 1, trimends=st.session_state.endpoints
    )

    st.session_state.depth = z

    st.write(":grey[Regrided mask...]")
    st.write(":green[All data regrided!]")

    st.write("No. of grid depth bins before regridding: ", np.shape(velocity)[1])
    st.write(
        "No. of grid depth bins after regridding: ",
        np.shape(st.session_state.velocity_regrid)[1],
    )
    fillplot_plotly(
        st.session_state.velocity_regrid[0, :, :], title="Regridded Velocity File"
    )
    fillplot_plotly(velocity[0, :, :], title="Original File")
    fillplot_plotly(
        st.session_state.mask_regrid, colorscale="greys", title="Regridded Mask File"
    )

    st.session_state.isGrid = True
    st.session_state.isGridSave = False


########### Save and Reset Mask ##############
st.header("Save & Reset Mask Data", divider="blue")

col1, col2 = st.columns([1, 1])
with col1:
    save_mask_button = st.button(label="Save Mask Data")
    if save_mask_button:
        if st.session_state.isGrid:
            st.session_state.profile_mask = st.session_state.mask_regrid
            st.session_state.isGridSave = True
        else:
            st.session_state.profile_mask = st.session_state.maskp
        st.session_state.isProfileMask = True
        st.session_state.isVelocityMask = False
        st.write(":green[Mask data saved]")
    else:
        st.write(":red[Mask data not saved]")
with col2:
    reset_mask_button = st.button("Reset mask data")
    if reset_mask_button:
        st.session_state.maskp = np.copy(st.session_state.orig_mask)
        st.write(":green[Mask data is reset to default]")
        st.session_state.isQCMask = False
        st.session_state.isProfileMask = False
        st.session_state.isGrid = False
        st.session_state.isGridSave = False
        st.session_state.isVelocityMask = False
