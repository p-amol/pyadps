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

# `maskp` holds the temporary changes in the page
# `profile_mask`
if "maskp" not in st.session_state:
    if "qc_mask" not in st.session_state:
        st.session_state.maskp = np.copy(st.session_state.orig_mask)
    else:
        st.session_state.maskp = np.copy(st.session_state.qc_mask)

#Giving infromations and warnings according to the state of masks.
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
        # Selecting the original mask file.
        if reset_selectbox == "Default":
            st.write("Default mask file selected")
            st.session_state.maskp = st.session_state.orig_mask
        elif reset_selectbox == "QC Test":
            st.write("QC Test mask file selected")
            st.session_state.maskp = st.session_state.qc_mask
        else:
            st.session_state.maskp = st.session_state.profile_mask
    elif("isTrimEnds" not in st.session_state):
        st.session_state.maskp = st.session_state.qc_mask
else:
    st.write(":orange[Creating a new mask file ...]")

# Creating a local copy of the mask.
mask = np.copy(st.session_state.maskp) 

# Load data
ds = st.session_state.ds
flobj = st.session_state.flead
vlobj = st.session_state.vlead
velocity = st.session_state.velocity
echo = st.session_state.echo
correlation = st.session_state.correlation
pgood = st.session_state.pgood
fdata = flobj.fleader
vdata = vlobj.vleader

# Settingup parameters for plotting graphs.
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
    if mask is not None:
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

n = 20
m = 20
if "update_mask" not in st.session_state:
    st.session_state.update_mask = False
    st.session_state.endpoints = None
    st.session_state.isTrimEnds = False
if "update_mask_cutbin" not in st.session_state:
    st.session_state.update_mask_cutbin = False
    st.session_state.isCutBins = False

st.header("Profile Test")

# Creating tabs for each actions
tab1, tab2, tab3, tab4 = st.tabs(["Trim Ends", "Cut Bins - Sidelobe", "Cut Bins - Manual", "Regriding"])

############## TRIM ENDS #################
with tab1:
    st.header("Trim Ends", divider="blue")
    ens_range = st.number_input("Change range", x[0], x[-1], 20)
    start_ens = st.slider("Deployment Ensembles", 0, ens_range, 0)
    end_ens = st.slider("Recovery Ensembles", x[-1] - ens_range, x[-1] + 1, x[-1] + 1)

    n = int(ens_range)

    if start_ens or end_ens:
        trim_ends(start_ens=start_ens, end_ens=end_ens, ens_range=n)
        # st.session_state.update_mask = False

    update_mask = st.button("Update mask data")
    if update_mask:
        if start_ens > 0:
            mask[:, :start_ens] = 1

        if end_ens <= x[-1]:
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
        st.session_state.isTrimEnds = True

    if not st.session_state.update_mask:
        st.write(":red[mask data not updated]")

    # Track button clicks
    if "cutBinsManual" not in st.session_state:
        st.session_state.cutBinsManual = False


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
        orientation = st.session_state.beam_direction
        st.write(f"The orientation is `{orientation}`.")
        beam = st.radio("Select beam", (1, 2, 3, 4), horizontal=True)
        beam = beam - 1
        st.session_state.beam = beam

    with right1:
        water_column_depth = 0
        with st.form(key="cutbin_form"):
            extra_cells = st.number_input("Additional Cells to Delete", 0, 10, 0)
            if orientation.lower() == 'down':
                water_column_depth = st.number_input("Enter water column depth (m): ", 0, 15000, 0) 
            cut_bins_mask = st.form_submit_button(label="Cut bins")
    

    left2, right2 = st.columns([1, 1])
    if "mask_temp" not in st.session_state:
        st.session_state.mask_temp = np.copy(mask)

    with left2:
        if cut_bins_mask:
            st.session_state.extra_cells = extra_cells
            st.session_state.mask_temp = side_lobe_beam_angle(ds, mask,
                                        orientation=orientation,
                                        water_column_depth=water_column_depth,
                                        extra_cells=extra_cells)
            fillplot_plotly(
                echo[beam, :, :],
                title="Echo Intensity (Masked)",
                maskdata=st.session_state.mask_temp,)
        else:
            if st.session_state.isCutBins:
                fillplot_plotly(
                echo[beam, :, :],
                title="Echo Intensity (Masked)",
                maskdata=st.session_state.mask_temp,)
            else:
                fillplot_plotly(echo[beam, :, :], maskdata=st.session_state.mask_temp, title="Echo Intensity")
    update_mask_cutbin = st.button("Update mask file after cutbin")
    if update_mask_cutbin:
        mask = np.copy(st.session_state.mask_temp)
        st.session_state.maskp = mask
        st.write(":green[mask file updated]")
        st.session_state.update_mask_cutbin = True
        st.session_state.isCutBins = True
    
    with right2:
        fillplot_plotly(mask, colorscale="greys", title="Mask Data ")
    if not st.session_state.update_mask_cutbin:
        st.write(":red[mask file not updated]")


########### CUT BINS: Manual #################
with tab3:
    st.header("Cut Bins: Manual", divider="blue")

    left3, right3 = st.columns([1, 2])
    with left3:
        # User selects beam (1-4)
        beam = st.radio("Select beam", (1, 2, 3, 4), horizontal=True,  key="beam_selection")
        beam_index = beam - 1
        st.subheader("Mask Selected Regions")
        with st.form(key="manual_cutbin_form"):
            st.write("Select the specific range of cells and ensembles to delete")

            # Input for selecting minimum and maximum cells
            min_cell = st.number_input("Min Cell", 0, int(flobj.field()["Cells"]), 0)
            max_cell = st.number_input("Max Cell", 0, int(flobj.field()["Cells"]), int(flobj.field()["Cells"]))

            # Input for selecting minimum and maximum ensembles
            min_ensemble = st.number_input("Min Ensemble", 0, int(flobj.ensembles), 0)
            max_ensemble = st.number_input("Max Ensemble", 0, int(flobj.ensembles), int(flobj.ensembles))

            # Submit button to apply the mask
            cut_bins_mask_manual = st.form_submit_button(label="Apply Manual Cut Bins")
        
        # Adding the new feature: Delete Single Cell or Ensemble
        st.subheader("Delete Specific Cell or Ensemble")

        # Step 1: User chooses between deleting a cell or an ensemble
        delete_option = st.radio("Select option to delete", ("Cell", "Ensemble"), horizontal=True)

        # Step 2: Display options based on user's choice
        if delete_option == "Cell":
            # Option to delete a specific cell across all ensembles
            with st.form(key="delete_cell_form"):
                st.write("Select a specific cell to delete across all ensembles")
                
                # Input for selecting a single cell
                cell = st.number_input("Cell", 0, int(flobj.field()["Cells"]), 0, key="single_cell")
                
                # Submit button to apply the mask for cell deletion
                delete_cell = st.form_submit_button(label="Delete Cell")

                if delete_cell:
                    mask[cell, :] = 1  # Mask the entire row for the selected cell
                    st.session_state.maskp = mask

        elif delete_option == "Ensemble":
            # Option to delete a specific ensemble across all cells
            with st.form(key="delete_ensemble_form"):
                st.write("Select a specific ensemble to delete across all cells")
                
                # Input for selecting a specific ensemble
                ensemble = st.number_input("Ensemble", 0, int(flobj.ensembles), 0, key="single_ensemble")
                
                # Submit button to apply the mask for ensemble deletion
                delete_ensemble = st.form_submit_button(label="Delete Ensemble")

                if delete_ensemble:
                    mask[:, ensemble-1] = 1  # Mask the entire column for the selected ensemble
                    st.session_state.maskp = mask

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
            ("Velocity", "Echo Intensity", "Correlation", "Percentage Good")
        )
        # Display the selected variable and beam
        selected_data = data_dict[variable][beam_index, :, :]
        if cut_bins_mask_manual:
            mask = np.copy(st.session_state.mask_temp)
            mask = manual_cut_bins(mask, min_cell, max_cell, min_ensemble, max_ensemble)
            st.session_state.maskp = np.copy(mask)
        fillplot_plotly(
            selected_data,
            title= variable + "(Masked Manually)",
            maskdata=mask,
        )
        # else:
        #     fillplot_plotly(selected_data, title=f"{variable}")
        fillplot_plotly(mask, colorscale="greys", title="Mask Data")


    # Layout with two columns
    col1, col2 = st.columns([1, 3])

    with col1:
                
        # Button to save mask data after manual cut bins, with unique key
        update_mask_cutbin = st.button("Update mask file after cutbin Manual", key="update_cutbin_button")
        if update_mask_cutbin:
            st.session_state.maskp = mask
            st.write(":green[mask file updated]")
            st.session_state.update_mask_cutbin = True
            #st.session_state.isCutBins = True
            st.session_state.cutBinsManual = True

        if not st.session_state.update_mask_cutbin:
            st.write(":red[mask file not updated]")

    with col2:
        # Button to reset the mask data, with unique key
        reset_mask_button = st.button("Reset mask data", key="reset_mask_button")
        if reset_mask_button:
            st.session_state.maskp = np.copy(st.session_state.orig_mask)
            st.write(":green[Mask data is reset to default]")
            st.session_state.isQCMask = False
            st.session_state.isProfileMask = False
            st.session_state.isGrid = False
            st.session_state.isGridSave = False
            st.session_state.isVelocityMask = False

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
        if st.session_state.beam_direction.lower() == "up":
            end_bin_option = st.radio(
                "Select the depth of last bin for regridding", ("Cell", "Surface", "Manual"), horizontal=True
            )
        else:
            end_bin_option = st.radio(
                "Select the depth of last bin for regridding", ("Cell", "Manual"), horizontal=True
            )

        st.session_state.end_bin_option = end_bin_option 
        st.write(f"You have selected: `{end_bin_option}`")

        if end_bin_option == "Manual":
            mean_depth = np.mean(st.session_state.vlead.vleader["Depth of Transducer"]) / 10
            mean_depth = round(mean_depth, 2)

            st.write(f"The transducer depth is {mean_depth} m. The value should not exceed the transducer depth")
            if st.session_state.beam_direction.lower() == "up":
                boundary = st.number_input("Enter the depth (m):", max_value=int(mean_depth), min_value=0)
            else:
                boundary = st.number_input("Enter the depth (m):", min_value=int(mean_depth))
        else:
            boundary = 0

        interpolate = st.radio("Choose interpolation method:", ("nearest", "linear", "cubic"))

        regrid_button = st.button(label="Regrid Data")
        if regrid_button:
            print(type(ds), ds)
            # st.write(st.session_state.endpoints)
            z, st.session_state.velocity_regrid = regrid3d(
                ds, velocity, -32768, 
                trimends=st.session_state.endpoints, 
                end_bin_option=st.session_state.end_bin_option, 
                orientation=st.session_state.beam_direction,
                method=interpolate,
                boundary_limit=boundary
            )
            st.write(":grey[Regrided velocity ...]")
            z, st.session_state.echo_regrid = regrid3d(
                ds, echo, -32768, 
                trimends=st.session_state.endpoints, 
                end_bin_option=st.session_state.end_bin_option, 
                orientation=st.session_state.beam_direction,
                method=interpolate,
                boundary_limit=boundary
            )
            st.write(":grey[Regrided echo intensity ...]")
            z, st.session_state.correlation_regrid = regrid3d(
                ds, correlation, -32768,
                trimends=st.session_state.endpoints, 
                end_bin_option=st.session_state.end_bin_option, 
                orientation=st.session_state.beam_direction,
                method=interpolate,
                boundary_limit=boundary
            )
            st.write(":grey[Regrided correlation...]")
            z, st.session_state.pgood_regrid = regrid3d(
                ds, pgood, -32768,
                trimends=st.session_state.endpoints, 
                end_bin_option=st.session_state.end_bin_option, 
                orientation=st.session_state.beam_direction,
                method=interpolate,
                boundary_limit=boundary
            )
            st.write(":grey[Regrided percent good...]")
            z, st.session_state.mask_regrid = regrid2d(
                ds, mask, 1,
                trimends=st.session_state.endpoints, 
                end_bin_option=st.session_state.end_bin_option, 
                orientation=st.session_state.beam_direction,
                method="nearest",
                boundary_limit=boundary
            )

            st.session_state.depth = z

            st.write(":grey[Regrided mask...]")
            st.write(":green[All data regrided!]")

            st.write("No. of grid depth bins before regridding: ", np.shape(velocity)[1])
            st.write(
                "No. of grid depth bins after regridding: ",
                np.shape(st.session_state.velocity_regrid)[1],
            )
            st.session_state.isGrid = True
            st.session_state.isGridSave = False

    with right4:
        if st.session_state.isGrid:
            fillplot_plotly(
                st.session_state.velocity_regrid[0, :, :], title="Regridded Velocity File",
            maskdata=st.session_state.mask_regrid)
            fillplot_plotly(
            st.session_state.mask_regrid, colorscale="greys", title="Regridded Mask File"
            )
        else:
            fillplot_plotly(velocity[0, :, :], maskdata=st.session_state.maskp, title="Original File")
        

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
        # Table summarizing changes
        changes_summary = pd.DataFrame(
            [
                ["Trim Ends", "True" if st.session_state.isTrimEnds else "False"],
                ["Cut Bins: Side Lobe Contamination", "True" if st.session_state.isCutBins else "False"],
                ["Cut Bins: Manual", "True" if st.session_state.cutBinsManual else "False"],
                ["Regrid Depth Cells", "True" if st.session_state.isGrid else "False"],
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
        styled_table = changes_summary.style.set_properties(**{"text-align": "center"})
        styled_table = styled_table.map(status_color_map, subset=["Status"])

        # Display the styled table
        st.write(styled_table.to_html(), unsafe_allow_html=True)
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
        st.session_state.isTrimEnds = False
        st.session_state.isCutBins = False
        st.session_state.cutBinsManual = False