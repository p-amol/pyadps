import numpy as np
import pandas as pd

# import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from utils.profile_test import side_lobe_beam_angle, manual_cut_bins
from utils.profile_test import regrid2d, regrid3d
from utils.signal_quality import default_mask

# If no file is uploaded, then give a warning
if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()


def reset_profiletest():
    # Reset Global Test
    st.session_state.isProfileTest = False
    # Reset Local Tests
    st.session_state.isTrimEndsCheck_PT = False
    st.session_state.isCutBinSideLobeCheck_PT = False
    st.session_state.isCutBinManualCheck_PT = False
    st.session_state.isRegridCheck_PT = False

    # Reset Page Return
    st.session_state.isQCPageReturn = False
    if not st.session_state.isQCTest:
        st.session_state.isSensorPageReturn = False

    # Reset Data
    st.session_state.echo_regrid = np.copy(st.session_state.echo)
    st.session_state.correlation_regrid = np.copy(st.session_state.correlation)
    st.session_state.pgood_regrid = np.copy(st.session_state.pgood)
    if st.session_state.isVelocityModifiedSound_ST:
        st.session_state.velocity_regrid = np.copy(st.session_state.velocity_sensor)
        st.session_state.velocity_temp = np.copy(st.session_state.velocity_sensor)
    else:
        st.session_state.velocity_regrid = np.copy(st.session_state.velocity)
        st.session_state.velocity_temp = np.copy(st.session_state.velocity)

    # Reset Mask Data
    if st.session_state.isQCTest:
        st.session_state.profile_mask_default = np.copy(st.session_state.qc_mask)
    elif st.session_state.isSensorTest:
        st.session_state.profile_mask_default = np.copy(st.session_state.sensor_mask)
    else:
        st.session_state.profile_mask_default = np.copy(st.session_state.orig_mask)

    mask = st.session_state.profile_mask_default
    st.session_state.profile_mask_temp = np.copy(mask)
    st.session_state.profile_mask = np.copy(mask)
    st.session_state.profile_mask_trimends = np.copy(mask)
    st.session_state.profile_mask_sidelobe = np.copy(mask)
    st.session_state.profile_mask_manual = np.copy(mask)
    st.session_state.profile_mask_regrid = np.copy(mask)

    st.session_state.sidelobe_displaymask = np.copy(mask)
    st.session_state.manual_displaymask = np.copy(mask)


def hard_reset(option):
    # Reset Global Test
    st.session_state.isProfileTest = False

    # Reset Local Tests
    st.session_state.isTrimEndsCheck_PT = False
    st.session_state.isCutBinSideLobeCheck_PT = False
    st.session_state.isCutBinManualCheck_PT = False
    st.session_state.isRegridCheck_PT = False

    # Reset Page Return
    st.session_state.isQCPageReturn = False
    if not st.session_state.isQCTest:
        st.session_state.isSensorPageReturn = False

    # Reset Data
    st.session_state.echo_regrid = np.copy(st.session_state.echo)
    st.session_state.correlation_regrid = np.copy(st.session_state.correlation)
    st.session_state.pgood_regrid = np.copy(st.session_state.pgood)
    if st.session_state.isVelocityModifiedSound_ST:
        st.session_state.velocity_regrid = np.copy(st.session_state.velocity_sensor)
        st.session_state.velocity_temp = np.copy(st.session_state.velocity_sensor)
    else:
        st.session_state.velocity_regrid = np.copy(st.session_state.velocity)
        st.session_state.velocity_temp = np.copy(st.session_state.velocity)

    # Reset Mask Data based on user options
    if option == "Sensor Test":
        st.session_state.profile_mask_default = np.copy(st.session_state.sensor_mask)
    if option == "QC Test":
        st.session_state.profile_mask_default = np.copy(st.session_state.qc_mask)
    elif option == "Default":
        st.session_state.profile_mask_default = np.copy(st.session_state.orig_mask)

    st.session_state.profile_mask = np.copy(st.session_state.profile_mask_default)
    st.session_state.profile_mask_temp = np.copy(st.session_state.profile_mask_default)
    st.session_state.profile_mask_trimends = np.copy(
        st.session_state.profile_mask_default
    )
    st.session_state.profile_mask_sidelobe = np.copy(
        st.session_state.profile_mask_default
    )
    st.session_state.profile_mask_manual = np.copy(
        st.session_state.profile_mask_default
    )
    st.session_state.profile_mask_regrid = np.copy(
        st.session_state.profile_mask_default
    )
    st.session_state.profile_mask_displaymask = np.copy(
        st.session_state.profile_mask_default
    )


# Load data
ds = st.session_state.ds
flobj = st.session_state.flead
vlobj = st.session_state.vlead
echo = st.session_state.echo
correlation = st.session_state.correlation
pgood = st.session_state.pgood
fdata = flobj.fleader
vdata = vlobj.vleader

# Setting up parameters for plotting graphs.
ensembles = st.session_state.head.ensembles
cells = flobj.field()["Cells"]
beams = flobj.field()["Beams"]
cell_size = flobj.field()["Depth Cell Len"]
bin1dist = flobj.field()["Bin 1 Dist"]
beam_angle = int(flobj.system_configuration()["Beam Angle"])

x = np.arange(0, ensembles, 1)
y = np.arange(0, cells, 1)

# Regrided data
# if "velocity_regrid" not in st.session_state:
#     st.session_state.echo_regrid = np.copy(echo)
#     st.session_state.velocity_regrid = np.copy(velocity)
#     st.session_state.correlation_regrid = np.copy(correlation)
#     st.session_state.pgood_regrid = np.copy(pgood)
#     st.session_state.mask_regrid = np.copy(mask)


# @st.cache_data
def fillplot_plotly(
    data, title="data", maskdata=None, missing=-32768, colorscale="balance"
):
    fig = FigureResampler(go.Figure())
    data = np.int32(data)
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
    if maskdata is not None:
        fig.add_trace(
            go.Heatmap(
                z=maskdata,
                x=x,
                y=y,
                colorscale="gray",
                hoverongaps=False,
                showscale=False,
                opacity=0.4,
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
    data = np.int32(data)
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


# Trim-end functions
def set_button_trimends():
    # Trim ends only modifies Mask
    mask = st.session_state.profile_mask_default
    if st.session_state.start_ens_PT > 0:
        mask[:, : st.session_state.start_ens_PT] = 1

    if st.session_state.end_ens_PT <= x[-1]:
        mask[:, st.session_state.end_ens_PT :] = 1

    st.session_state.profile_mask_trimends = np.copy(mask)
    st.session_state.profile_mask_temp = np.copy(mask)
    st.session_state.isTrimEndsCheck_PT = True


def reset_button_trimends():
    # Trim ends only modifies Mask
    st.session_state.isTrimEndsCheck_PT = False
    st.session_state.profile_mask_trimends = st.session_state.profile_mask_default
    st.session_state.profile_mask_temp = st.session_state.profile_mask_default


# Side-lobe functions
def set_button_apply_sidelobe():
    inmask = np.copy(st.session_state.profile_mask_temp)
    transdepth = st.session_state.depth
    water_column_depth = st.session_state.water_depth_PT
    extra_cells = st.session_state.extra_cell_PT
    mask = side_lobe_beam_angle(
        transdepth,
        inmask,
        orientation=orientation,
        water_column_depth=water_column_depth,
        extra_cells=extra_cells,
        cells=cells,
        cell_size=cell_size,
        bin1dist=bin1dist,
        beam_angle=beam_angle,
    )
    st.session_state.sidelobe_displaymask = np.copy(mask)


def set_button_sidelobe():
    st.session_state.isCutBinSideLobeCheck_PT = True
    st.session_state.profile_mask_temp = np.copy(st.session_state.sidelobe_displaymask)
    st.session_state.profile_mask_sidelobe = np.copy(st.session_state.profile_mask_temp)


def reset_button_sidelobe():
    st.session_state.isCutBinSideLobeCheck_PT = False
    if st.session_state.isTrimEndsCheck_PT:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_trimends
    else:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_default

    st.session_state.profile_mask_sidelobe = np.copy(st.session_state.profile_mask_temp)
    st.session_state.sidelobe_displaymask = np.copy(st.session_state.profile_mask_temp)


# Cutbins Manual Functions
def set_button_apply_mask_region():
    mask = np.copy(st.session_state.profile_mask_temp)

    min_cell = st.session_state.profile_min_cell
    max_cell = st.session_state.profile_max_cell
    min_ensemble = st.session_state.profile_min_ensemble
    max_ensemble = st.session_state.profile_max_ensemble

    mask = manual_cut_bins(mask, min_cell, max_cell, min_ensemble, max_ensemble)
    st.session_state.manual_displaymask = np.copy(mask)
    st.session_state.profile_mask_temp = np.copy(mask)
    st.session_state.profile_mask_manual = np.copy(mask)

    st.session_state.isCutBinManualCheck_PT = True


def set_button_mask_region():
    st.session_state.isCutBinManualCheck_PT = True
    st.session_state.profile_mask_manual = np.copy(st.session_state.profile_mask_temp)


def set_button_delete_cell():
    cell = st.session_state.profile_delete_cell
    mask = st.session_state.profile_mask_temp
    mask[cell, :] = 1  # Mask the entire row for the selected cell
    st.session_state.profile_mask_temp = np.copy(mask)
    st.session_state.profile_mask_manual = np.copy(mask)

    st.session_state.isCutBinManualCheck_PT = True


def set_button_delete_ensemble():
    ensemble = st.session_state.profile_delete_ensemble
    mask = st.session_state.profile_mask_temp
    mask[:, ensemble - 1] = 1  # Mask the entire column for the selected ensemble
    st.session_state.profile_mask_temp = np.copy(mask)
    st.session_state.profile_mask_manual = np.copy(mask)
    st.session_state.isCutBinManualCheck_PT = True


def reset_button_mask_manual():
    st.session_state.isCutBinManualCheck_PT = False
    # st.session_state.isCutBinManualOn = False

    if st.session_state.isCutBinSideLobeCheck_PT:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_sidelobe
    elif st.session_state.isTrimEndsCheck_PT:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_trimends
    else:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_default

    # Copy to local mask
    st.session_state.profile_mask_manual = np.copy(st.session_state.profile_mask_temp)


# Regrid functions
def reset_button_regrid():
    st.session_state.isRegridCheck_PT = False

    # Reset to previous state
    if st.session_state.isCutBinManualCheck_PT:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_manual
    elif st.session_state.isCutBinSideLobeCheck_PT:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_sidelobe
    elif st.session_state.isTrimEndsCheck_PT:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_trimends
    else:
        st.session_state.profile_mask_temp = st.session_state.profile_mask_default

    # Copy to local mask
    st.session_state.profile_mask_regrid = np.copy(st.session_state.profile_mask_temp)

    # Reset Data
    if st.session_state.isVelocityModifiedSound_ST:
        st.session_state.velocity_regrid = np.copy(st.session_state.velocity_sensor)
    else:
        st.session_state.velocity_regrid = np.copy(st.session_state.velocity)
    st.session_state.echo_regrid = np.copy(st.session_state.echo)
    st.session_state.velocity_regrid = np.copy(st.session_state.velocity)
    st.session_state.correlation_regrid = np.copy(st.session_state.correlation)
    st.session_state.pgood_regrid = np.copy(st.session_state.pgood)


def save_profiletest():
    if st.session_state.isRegridCheck_PT:
        st.session_state.profile_mask = st.session_state.profile_mask_regrid
    else:
        st.session_state.profile_mask = st.session_state.profile_mask_temp
    # st.session_state.velocity_regrid = np.copy(st.session_state.velocity_temp)
    st.session_state.isProfileTest = True
    st.session_state.isVelocityTest = False

    st.session_state.isSensorPageReturn = True
    st.session_state.isQCPageReturn = True


# Giving infromations and warnings according to the state of masks.
if st.session_state.isQCTest or st.session_state.isSensorTest:
    st.write(":grey[Working on a saved mask file ...]")
    if st.session_state.isProfilePageReturn:
        st.write(
            ":orange[Warning: Profile test already completed. Reset to change settings.]"
        )
        reset_selectbox = st.selectbox(
            "Choose reset option",
            ("Sensor Test", "QC Test", "Default"),
            index=None,
            placeholder="Reset mask to ...",
        )
        # Selecting the original mask file.
        if reset_selectbox == "Default":
            st.write("Default mask file selected")
        elif reset_selectbox == "Sensor Test":
            st.write("Sensor Test mask file selected")
        elif reset_selectbox == "QC Test":
            st.write("QC Test mask file selected")
        if reset_selectbox is not None:
            hard_reset(reset_selectbox)
    elif st.session_state.isFirstProfileVisit:
        reset_profiletest()
        st.session_state.isFirstProfileVisit = False
else:
    if st.session_state.isFirstProfileVisit:
        reset_profiletest()
        st.session_state.isFirstProfileVisit = False
    st.write(":orange[Creating a new mask file ...]")

st.header("Profile Test")
# Creating tabs for each actions
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Trim Ends",
        "Cut Bins - Sidelobe",
        "Cut Bins - Manual",
        "Regriding",
        "Save & Reset",
    ]
)


############## TRIM ENDS #################
with tab1:
    st.header("Trim Ends", divider="blue")
    x1 = np.arange(0, ensembles, 1)
    st.session_state.ens_range = st.number_input("Change range", x[0], x[-1], 20)
    st.session_state.start_ens = st.slider(
        "Deployment Ensembles", 0, st.session_state.ens_range, 0
    )
    st.session_state.end_ens = st.slider(
        "Recovery Ensembles",
        x1[-1] - st.session_state.ens_range,
        x1[-1] + 1,
        x1[-1] + 1,
    )

    if st.session_state.start_ens or st.session_state.end_ens:
        trim_ends(
            start_ens=int(st.session_state.start_ens),
            end_ens=int(st.session_state.end_ens),
            ens_range=int(st.session_state.ens_range),
        )

    st.session_state.start_ens_PT = st.session_state.start_ens
    st.session_state.end_ens_PT = st.session_state.end_ens
    st.session_state.trimends_endpoints = np.array(
        [st.session_state.start_ens_PT, st.session_state.end_ens_PT]
    )

    left_te, right_te = st.columns([1, 1])
    with left_te:
        trimends_mask_button = st.button("Trim Ends", on_click=set_button_trimends)

        # Display output
        if trimends_mask_button:
            st.success("Mask data updated")
            st.write("Trim End Points", st.session_state.trimends_endpoints)
        else:
            st.write(":red[mask data not updated]")

    with right_te:
        trimends_reset_button = st.button(
            "Reset Trim Ends", on_click=reset_button_trimends
        )
        if trimends_reset_button:
            st.success("Mask data reset to default.")


############  CUT BINS (SIDE LOBE) ############################
with tab2:
    st.header("Cut Bins: Side Lobe Contamination", divider="blue")
    st.write(
        """
    The side lobe echos from hard surface such as sea surface or bottom of the ocean can contaminate
    data closer to this region. The data closer to the surface or bottom can be removed using 
    the relation between beam angle and the thickness of the contaminated layer.
    """
    )

    left1, right1 = st.columns([1, 1])

    with left1:
        orientation = st.session_state.beam_direction_QCT
        st.write(f"The orientation is `{orientation}`.")
        beam = st.radio("Select beam", (1, 2, 3, 4), horizontal=True)
        beam = beam - 1
        st.session_state.beam = beam

    with right1:
        st.session_state.water_depth_PT = 0

        st.session_state.extra_cell_PT = st.number_input(
            "Additional Cells to Delete", 0, 10, 0
        )
        if orientation.lower() == "down":
            water_column_depth = st.number_input(
                "Enter water column depth (m): ", 0, 15000, 0
            )
        side_lobe_button = st.button(
            label="Cut bins", on_click=set_button_apply_sidelobe
        )

    left2, right2 = st.columns([1, 1])

    with left2:
        # if st.session_state.isCutBinSideLobeCheck_PT:
        #     fillplot_plotly(
        #         echo[beam, :, :],
        #         title="Echo Intensity (Masked)",
        #         maskdata=st.session_state.profile_mask_temp,
        #     )
        # else:
        fillplot_plotly(
            echo[beam, :, :],
            maskdata=st.session_state.sidelobe_displaymask,
            title="Echo Intensity",
        )

    with right2:
        fillplot_plotly(
            st.session_state.sidelobe_displaymask,
            colorscale="greys",
            title="Mask Data ",
        )

    left3sl, right3sl = st.columns([1, 1])
    with left3sl:
        update_mask_cutbin_button = st.button(
            "Update Mask (Side Lobe)", on_click=set_button_sidelobe
        )
        if update_mask_cutbin_button:
            st.success("Mask file updated")

    with right3sl:
        reset_mask_cutbin_button = st.button(
            "Reset Mask (Side lobe)", on_click=reset_button_sidelobe
        )
        if reset_mask_cutbin_button:
            st.info("Mask File Reset (Side Lobe)")

    if not update_mask_cutbin_button:
        st.write(":red[mask file not updated]")


########### CUT BINS: Manual #################
with tab3:
    st.header("Cut Bins: Manual", divider="blue")

    left3, right3 = st.columns([1, 2])
    with left3:
        # User selects beam (1-4)
        beam = st.radio(
            "Select beam", (1, 2, 3, 4), horizontal=True, key="beam_selection"
        )
        beam_index = beam - 1
        st.subheader("Mask Selected Regions")
        # with st.form(key="manual_cutbin_form"):
        st.write("Select the specific range of cells and ensembles to delete")

        # Input for selecting minimum and maximum cells
        st.session_state.profile_min_cell = st.number_input(
            "Min Cell", 0, int(flobj.field()["Cells"]), 0
        )
        st.session_state.profile_max_cell = st.number_input(
            "Max Cell", 0, int(flobj.field()["Cells"]), int(flobj.field()["Cells"])
        )

        st.write(st.session_state.profile_max_cell)
        # Input for selecting minimum and maximum ensembles
        st.session_state.profile_min_ensemble = st.number_input(
            "Min Ensemble", 0, int(flobj.ensembles), 0
        )
        st.session_state.profile_max_ensemble = st.number_input(
            "Max Ensemble", 0, int(flobj.ensembles), int(flobj.ensembles)
        )

        # Submit button to apply the mask
        cut_bins_mask_manual = st.button(label="Apply Manual Cut Bins")
        if cut_bins_mask_manual:
            set_button_apply_mask_region()

        # Adding the new feature: Delete Single Cell or Ensemble
        st.subheader("Delete Specific Cell or Ensemble")

        # Step 1: User chooses between deleting a cell or an ensemble
        delete_option = st.radio(
            "Select option to delete", ("Cell", "Ensemble"), horizontal=True
        )

        # Step 2: Display options based on user's choice
        if delete_option == "Cell":
            # Option to delete a specific cell across all ensembles
            st.write("Select a specific cell to delete across all ensembles")

            # Input for selecting a single cell
            st.session_state.profile_delete_cell = st.number_input(
                "Cell", 0, int(flobj.field()["Cells"]), 0, key="single_cell"
            )

            # Submit button to apply the mask for cell deletion
            delete_cell_button = st.button(
                label="Delete Cell", on_click=set_button_delete_cell
            )

            if delete_cell_button:
                st.write("Deleted cell: ", st.session_state.profile_delete_cell)

        if delete_option == "Ensemble":
            # Option to delete a specific ensemble across all cells
            st.write("Select a specific ensemble to delete across all cells")

            # Input for selecting a specific ensemble
            st.session_state.profile_delete_ensemble = st.number_input(
                "Ensemble", 0, int(flobj.ensembles), 0, key="single_ensemble"
            )

            # Submit button to apply the mask for ensemble deletion
            delete_ensemble_button = st.button(
                label="Delete Ensemble", on_click=set_button_delete_ensemble
            )

            if delete_ensemble_button:
                st.write("Deleted Ensemble: ", st.session_state.profile_delete_ensemble)

    velocity = st.session_state.velocity_temp
    # Map variable selection to corresponding data
    data_dict = {
        "Velocity": velocity,
        "Echo Intensity": echo,
        "Correlation": correlation,
        "Percentage Good": pgood,
    }

    with right3:
        # Selection of variable (Velocity, Echo Intensity, etc.)
        variable = st.selectbox(
            "Select Variable to Display",
            ("Velocity", "Echo Intensity", "Correlation", "Percentage Good"),
        )
        # Display the selected variable and beam
        selected_data = data_dict[variable][beam_index, :, :]
        fillplot_plotly(
            selected_data,
            title=variable + "(Masked Manually)",
            maskdata=st.session_state.profile_mask_temp,
        )
        # else:
        #     fillplot_plotly(selected_data, title=f"{variable}")
        fillplot_plotly(
            st.session_state.profile_mask_temp,
            colorscale="greys",
            title="Mask Data",
        )

    # Layout with two columns
    col1, col2 = st.columns([1, 3])

    # Button to reset the mask data, with unique key
    reset_mask_button = st.button(
        "Reset Cut Bins Manual",
        key="reset_mask_button",
        on_click=reset_button_mask_manual,
    )
    if reset_mask_button:
        st.info("Cut Bins Manual Reset. Mask data is changed to previous state.")

############ REGRID ###########################################
with tab4:
    st.header("Regrid Depth Cells", divider="blue")

    st.write(
        """
    When the ADCP buoy has vertical oscillations (greater than depth cell size), 
    the depth bins has to be regridded based on the pressure sensor data. The data
    can be regrided either till the surface or till the last bin. 
    If the `Cell` option is selected, ensure that the end data are trimmed.
    Manual option permits choosing the end cell depth.
    """
    )

    left4, right4 = st.columns([1, 3])

    with left4:
        if st.session_state.beam_direction_QCT.lower() == "up":
            end_cell_option = st.radio(
                "Select the depth of last bin for regridding",
                ("Cell", "Surface", "Manual"),
                horizontal=True,
            )
        else:
            end_cell_option = st.radio(
                "Select the depth of last bin for regridding",
                ("Cell", "Manual"),
                horizontal=True,
            )

        st.session_state.end_cell_option_PT = end_cell_option
        st.write(f"You have selected: `{end_cell_option}`")

        if end_cell_option == "Manual":
            mean_depth = (
                np.mean(st.session_state.vlead.vleader["Depth of Transducer"]) / 10
            )
            mean_depth = round(mean_depth, 2)

            st.write(
                f"The transducer depth is {mean_depth} m. The value should not exceed the transducer depth"
            )
            if st.session_state.beam_direction_QCT.lower() == "up":
                boundary = st.number_input(
                    "Enter the depth (m):", max_value=int(mean_depth), min_value=0
                )
            else:
                boundary = st.number_input(
                    "Enter the depth (m):", min_value=int(mean_depth)
                )
        else:
            boundary = 0

        st.session_state.interpolate_PT = st.radio(
            "Choose interpolation method:", ("nearest", "linear", "cubic")
        )
        st.session_state.manualdepth_PT = boundary

        progress_text = "Regridding in progress. Please wait."
        grid_bar = st.progress(0, text=progress_text)

        regrid_button = st.button(label="Regrid Data")
        if regrid_button:
            transdepth = st.session_state.depth
            z, st.session_state.velocity_regrid = regrid3d(
                transdepth,
                st.session_state.velocity_temp,
                -32768,
                trimends=st.session_state.trimends_endpoints,
                end_cell_option=st.session_state.end_cell_option_PT,
                orientation=st.session_state.beam_direction_QCT,
                method=st.session_state.interpolate_PT,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beams=beams,
            )
            grid_bar.progress(20, text=progress_text)
            st.write(":grey[Regrided velocity ...]")
            z, st.session_state.echo_regrid = regrid3d(
                transdepth,
                echo,
                -32768,
                trimends=st.session_state.trimends_endpoints,
                end_cell_option=st.session_state.end_cell_option_PT,
                orientation=st.session_state.beam_direction_QCT,
                method=st.session_state.interpolate_PT,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beams=beams,
            )
            grid_bar.progress(40, text=progress_text)
            st.write(":grey[Regrided echo intensity ...]")
            z, st.session_state.correlation_regrid = regrid3d(
                transdepth,
                correlation,
                -32768,
                trimends=st.session_state.trimends_endpoints,
                end_cell_option=st.session_state.end_cell_option_PT,
                orientation=st.session_state.beam_direction_QCT,
                method=st.session_state.interpolate_PT,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beams=beams,
            )
            grid_bar.progress(60, text=progress_text)
            st.write(":grey[Regrided correlation...]")
            z, st.session_state.pgood_regrid = regrid3d(
                transdepth,
                pgood,
                -32768,
                trimends=st.session_state.trimends_endpoints,
                end_cell_option=st.session_state.end_cell_option_PT,
                orientation=st.session_state.beam_direction_QCT,
                method=st.session_state.interpolate_PT,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beams=beams,
            )
            grid_bar.progress(80, text=progress_text)
            st.write(":grey[Regrided percent good...]")

            z, st.session_state.profile_mask_regrid = regrid2d(
                transdepth,
                st.session_state.profile_mask_temp,
                1,
                trimends=st.session_state.trimends_endpoints,
                end_cell_option=st.session_state.end_cell_option_PT,
                orientation=st.session_state.beam_direction_QCT,
                method="nearest",
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
            )

            grid_bar.progress(99, text=progress_text)
            st.write(":grey[Regrided mask...]")

            st.session_state.depth_axis = z
            st.write(":grey[New depth axis created...]")

            grid_bar.progress(100, text="Completed")
            st.write(":green[All data regrided!]")

            st.write(
                "No. of grid depth bins before regridding: ", np.shape(velocity)[1]
            )
            st.write(
                "No. of grid depth bins after regridding: ",
                np.shape(st.session_state.velocity_regrid)[1],
            )
            st.session_state.isRegridCheck_PT = True

        regrid_reset_button = st.button(
            "Reset Regrid Test", on_click=reset_button_regrid
        )

        if regrid_reset_button:
            st.info("Data Reset")

    with right4:
        if st.session_state.isRegridCheck_PT:
            fillplot_plotly(
                st.session_state.velocity_regrid[0, :, :],
                title="Regridded Velocity File",
                maskdata=st.session_state.profile_mask_regrid,
            )
            fillplot_plotly(
                st.session_state.profile_mask_regrid,
                colorscale="greys",
                title="Regridded Mask File",
            )
        else:
            fillplot_plotly(
                velocity[0, :, :],
                maskdata=st.session_state.profile_mask_temp,
                title="Original File",
            )


########### Save and Reset Mask ##############
with tab5:
    st.header("Save & Reset Mask Data", divider="blue")

    col1, col2 = st.columns([1, 1])
    with col1:
        save_mask_button = st.button(label="Save Mask Data", on_click=save_profiletest)
        if save_mask_button:
            st.success("Mask data saved")
            # Table summarizing changes
            changes_summary = pd.DataFrame(
                [
                    [
                        "Trim Ends",
                        "True" if st.session_state.isTrimEndsCheck_PT else "False",
                    ],
                    [
                        "Cut Bins: Side Lobe Contamination",
                        "True"
                        if st.session_state.isCutBinSideLobeCheck_PT
                        else "False",
                    ],
                    [
                        "Cut Bins: Manual",
                        "True" if st.session_state.isCutBinManualCheck_PT else "False",
                    ],
                    [
                        "Regrid Depth Cells",
                        "True" if st.session_state.isRegridCheck_PT else "False",
                    ],
                ],
                columns=["Parameter", "Status"],
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
            styled_table = changes_summary.style.set_properties(
                **{"text-align": "center"}
            )
            styled_table = styled_table.map(status_color_map, subset=["Status"])

            # Display the styled table
            st.write(styled_table.to_html(), unsafe_allow_html=True)
        else:
            st.write(":red[Mask data not saved]")
    with col2:
        reset_mask_button = st.button("Reset mask data", on_click=reset_profiletest)
        if reset_mask_button:
            st.info("Mask data is reset to default")
