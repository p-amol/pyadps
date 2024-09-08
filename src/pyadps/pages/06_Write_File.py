import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import utils.writenc as wr
from plotly_resampler import FigureResampler

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()


if st.session_state.isVelocityMask:
    st.session_state.final_mask = st.session_state.velocity_mask
    st.session_state.final_velocity = st.session_state.veltest_velocity
    if st.session_state.isGridSave:
        st.session_state.final_echo = st.session_state.echo_regrid
        st.session_state.final_correlation = st.session_state.correlation_regrid
        st.session_state.final_pgood = st.session_state.pgood_regrid
    else:
        st.session_state.final_echo = st.session_state.echo
        st.session_state.final_correlation = st.session_state.correlation
        st.session_state.final_pgood = st.session_state.pgood
else:
    if st.session_state.isGridSave:
        st.session_state.final_mask = st.session_state.mask_regrid
        st.session_state.final_velocity = st.session_state.velocity_regrid
        st.session_state.final_echo = st.session_state.echo_regrid
        st.session_state.final_correlation = st.session_state.correlation_regrid
        st.session_state.final_pgood = st.session_state.pgood_regrid
    else:
        if st.session_state.isProfileMask:
            st.session_state.final_mask = st.session_state.profile_mask
        elif st.session_state.isQCMask:
            st.session_state.final_mask = st.session_state.qc_mask
        else:
            st.session_state.final_mask = st.session_state.orig_mask
        st.session_state.final_velocity = st.session_state.velocity
        st.session_state.final_echo = st.session_state.echo
        st.session_state.final_correlation = st.session_state.correlation
        st.session_state.final_pgood = st.session_state.pgood


if "depth" not in st.session_state:
    st.session_state.isGrid = False


@st.cache_data
def file_write(filename="processed_file.nc"):
    tempdirname = tempfile.TemporaryDirectory(delete=False)
    outfilepath = tempdirname.name + "/" + filename
    return outfilepath


# If the data is not regrided based on pressure sensor. Use the mean depth
if not st.session_state.isGrid:
    st.write(":red[WARNING!]")
    st.write(
        "Data not regrided. Using the mean transducer depth to calculate the depth axis."
    )
    mean_depth = np.mean(st.session_state.vlead.vleader["Depth of Transducer"]) / 10
    mean_depth = np.trunc(mean_depth)
    st.write(f"Mean depth of the transducer is `{mean_depth}`")
    cells = st.session_state.flead.field()["Cells"]
    cell_size = st.session_state.flead.field()["Depth Cell Len"] / 100
    bin1dist = st.session_state.flead.field()["Bin 1 Dist"] / 100
    max_depth = mean_depth - bin1dist
    min_depth = max_depth - cells * cell_size
    z = np.arange(-1 * max_depth, -1 * min_depth, cell_size)
    st.session_state.final_depth = z
else:
    st.session_state.final_depth = st.session_state.depth


@st.cache_data
def fillplot_plotly(
    x, y, data, maskdata, colorscale="balance", title="Data", mask=False
):
    fig = FigureResampler(go.Figure())
    if mask:
        data1 = np.where(maskdata == 1, np.nan, data)
    else:
        data1 = np.where(data == -32768, np.nan, data)

    fig.add_trace(
        go.Heatmap(
            z=data1[:, 0:-1],
            x=x,
            y=y,
            colorscale=colorscale,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        xaxis=dict(showline=True, mirror=True),
        yaxis=dict(showline=True, mirror=True),
        title_text=title,
    )
    st.plotly_chart(fig)


def call_plot(varname, beam, mask=False):
    if varname == "Velocity":
        fillplot_plotly(
            st.session_state.date,
            st.session_state.final_depth,
            st.session_state.final_velocity[beam - 1, :, :],
            st.session_state.final_mask,
            title=varname,
            mask=mask,
        )
    elif varname == "Echo":
        fillplot_plotly(
            st.session_state.date,
            st.session_state.final_depth,
            st.session_state.final_echo[beam - 1, :, :],
            st.session_state.final_mask,
            title=varname,
            mask=mask,
        )
    elif varname == "Correlation":
        fillplot_plotly(
            st.session_state.date,
            st.session_state.final_depth,
            st.session_state.final_correlation[beam - 1, :, :],
            st.session_state.final_mask,
            title=varname,
            mask=mask,
        )
    elif varname == "Percent Good":
        fillplot_plotly(
            st.session_state.date,
            st.session_state.final_depth,
            st.session_state.final_pgood[beam - 1, :, :],
            st.session_state.final_mask,
            title=varname,
            mask=mask,
        )


st.header("View Processed Data", divider="blue")
var_option = st.selectbox(
    "Select a data type", ("Velocity", "Echo", "Correlation", "Percent Good")
)
beam = st.radio("Select beam", (1, 2, 3, 4), horizontal=True)

mask_radio = st.radio("Apply Mask", ("Yes", "No"), horizontal=True)
plot_button = st.button("Plot Processed Data")
if plot_button:
    if mask_radio == "Yes":
        call_plot(var_option, beam, mask=True)
    elif mask_radio == "No":
        call_plot(var_option, beam, mask=False)


st.header("Write Data", divider="blue")

mask_data_radio = st.radio("Do you want to mask the final data?", ("Yes", "No"))

if mask_data_radio == "Yes":
    mask = st.session_state.final_mask
    st.session_state.write_velocity = np.copy(st.session_state.final_velocity)
    st.session_state.write_velocity[:, mask == 1] = -32768
else:
    st.session_state.write_velocity = np.copy(st.session_state.final_velocity)


file_type_radio = st.radio("Select output file format:", ("NetCDF", "CSV"))

download_button = st.button("Generate Processed files")


if download_button:
    st.session_state.processed_filename = file_write()
    st.write(":grey[Processed file created. Click the download button.]")
    st.write(st.session_state.processed_filename)
    depth = np.trunc(st.session_state.final_depth)
    if file_type_radio == "NetCDF":
        wr.finalnc(
            st.session_state.processed_filename,
            depth,
            st.session_state.date,
            st.session_state.write_velocity,
        )

        with open(st.session_state.processed_filename, "rb") as file:
            st.download_button(
                label="Download NetCDF File",
                data=file,
                file_name="processed_file.nc",
            )

    if file_type_radio == "CSV":
        udf = pd.DataFrame(
            st.session_state.write_velocity[0, :, :].T,
            index=st.session_state.date,
            columns=-1 * depth,
        )
        vdf = pd.DataFrame(
            st.session_state.write_velocity[1, :, :].T,
            index=st.session_state.date,
            columns=-1 * depth,
        )
        wdf = pd.DataFrame(
            st.session_state.write_velocity[2, :, :].T,
            index=st.session_state.date,
            columns=-1 * depth,
        )
        ucsv = udf.to_csv().encode("utf-8")
        vcsv = vdf.to_csv().encode("utf-8")
        wcsv = wdf.to_csv().encode("utf-8")
        st.download_button(
            label="Download Zonal Velocity File",
            data=ucsv,
            file_name="zonal_velocity.csv",
            mime="text/csf",
        )
        st.download_button(
            label="Download Meridional Velocity File",
            data=vcsv,
            file_name="meridional_velocity.csv",
            mime="text/csf",
        )
        st.download_button(
            label="Download Vertical Velocity File",
            data=vcsv,
            file_name="vertical_velocity.csv",
            mime="text/csf",
        )
