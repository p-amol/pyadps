import configparser
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

if "fname" not in st.session_state:
    st.session_state.fname = "No file selected"

if "rawfilename" not in st.session_state:
    st.session_state.rawfilename = "rawfile.nc"

if "vleadfilename" not in st.session_state:
    st.session_state.vleadfilename = "vlead.nc"


# Check if attributes exist in session state
if "attributes" not in st.session_state:
    st.session_state.attributes = {}

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


if "depth_axis" not in st.session_state:
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
    # mean_depth = np.mean(st.session_state.vlead.vleader["Depth of Transducer"]) / 10
    mean_depth = np.mean(st.session_state.depth)/10
    mean_depth = np.trunc(mean_depth)
    st.write(f"Mean depth of the transducer is `{mean_depth}`")
    cells = st.session_state.flead.field()["Cells"]
    cell_size = st.session_state.flead.field()["Depth Cell Len"] / 100
    bin1dist = st.session_state.flead.field()["Bin 1 Dist"] / 100
    max_depth = mean_depth - bin1dist
    min_depth = max_depth - cells * cell_size
    z = np.arange(-1 * max_depth, -1 * min_depth, cell_size)
    st.session_state.final_depth_axis = z
else:
    st.session_state.final_depth_axis = st.session_state.depth_axis


# Functions for plotting
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
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig)


def call_plot(varname, beam, mask=False):
    if varname == "Velocity":
        fillplot_plotly(
            st.session_state.date,
            st.session_state.final_depth_axis,
            st.session_state.final_velocity[beam - 1, :, :],
            st.session_state.final_mask,
            title=varname,
            mask=mask,
        )
    elif varname == "Echo":
        fillplot_plotly(
            st.session_state.date,
            st.session_state.final_depth_axis,
            st.session_state.final_echo[beam - 1, :, :],
            st.session_state.final_mask,
            title=varname,
            mask=mask,
        )
    elif varname == "Correlation":
        fillplot_plotly(
            st.session_state.date,
            st.session_state.final_depth_axis,
            st.session_state.final_correlation[beam - 1, :, :],
            st.session_state.final_mask,
            title=varname,
            mask=mask,
        )
    elif varname == "Percent Good":
        fillplot_plotly(
            st.session_state.date,
            st.session_state.final_depth_axis,
            st.session_state.final_pgood[beam - 1, :, :],
            st.session_state.final_mask,
            title=varname,
            mask=mask,
        )


# Option to View Processed Data
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


# Option to Write Processed Data
st.header("Write Data", divider="blue")

mask_data_radio = st.radio("Do you want to mask the final data?", ("Yes", "No"))

if mask_data_radio == "Yes":
    mask = st.session_state.final_mask
    st.session_state.write_velocity = np.copy(st.session_state.final_velocity)
    st.session_state.write_velocity[:, mask == 1] = -32768
else:
    st.session_state.write_velocity = np.copy(st.session_state.final_velocity)


file_type_radio = st.radio("Select output file format:", ("NetCDF", "CSV"))

if file_type_radio == "NetCDF":
    add_attr_button = st.checkbox("Add attributes to NetCDF file")

    if add_attr_button:
        st.write("### Modify Attributes")

        # Create two-column layout for attributes
        col1, col2 = st.columns(2)

        with col1:
            # Display attributes in the first column
            for key in [
                "Cruise_No.",
                "Ship_Name",
                "Project_No.",
                "Water_Depth_m",
                "Deployment_Depth_m",
                "Deployment_Date",
                "Recovery_Date",
            ]:
                if key in st.session_state.attributes:
                    st.session_state.attributes[key] = st.text_input(
                        key, value=st.session_state.attributes[key]
                    )
                else:
                    st.session_state.attributes[key] = st.text_input(key)

        with col2:
            # Display attributes in the second column
            for key in [
                "Latitude",
                "Longitude",
                "Platform_Type",
                "Participants",
                "File_created_by",
                "Contact",
                "Comments",
            ]:
                if key in st.session_state.attributes:
                    st.session_state.attributes[key] = st.text_input(
                        key, value=st.session_state.attributes[key]
                    )
                else:
                    st.session_state.attributes[key] = st.text_input(key)

download_button = st.button("Generate Processed files")

if download_button:
    st.session_state.processed_filename = file_write()
    st.write(":grey[Processed file created. Click the download button.]")
#    st.write(st.session_state.processed_filename)
    depth_axis = np.trunc(st.session_state.final_depth_axis)
    final_mask = st.session_state.final_mask

    if file_type_radio == "NetCDF":
        if add_attr_button and st.session_state.attributes:
            # Generate file with attributes
            wr.finalnc(
                st.session_state.processed_filename,
                depth_axis,
                final_mask,
                st.session_state.date,
                st.session_state.write_velocity,
                attributes=st.session_state.attributes,  # Pass edited attributes
            )
        else:
            # Generate file without attributes
            wr.finalnc(
                st.session_state.processed_filename,
                depth_axis,
                final_mask,
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
            columns=-1 * depth_axis,
        )
        vdf = pd.DataFrame(
            st.session_state.write_velocity[1, :, :].T,
            index=st.session_state.date,
            columns=-1 * depth_axis,
        )
        wdf = pd.DataFrame(
            st.session_state.write_velocity[2, :, :].T,
            index=st.session_state.date,
            columns=-1 * depth_axis,
        )
        ucsv = udf.to_csv().encode("utf-8")
        vcsv = vdf.to_csv().encode("utf-8")
        wcsv = wdf.to_csv().encode("utf-8")
        csv_mask = pd.DataFrame(st.session_state.final_mask.T).to_csv().encode("utf-8")
        st.download_button(
            label="Download Zonal Velocity File (CSV)",
            data=ucsv,
            file_name="zonal_velocity.csv",
            mime="text/csf",
        )
        st.download_button(
            label="Download Meridional Velocity File (CSV)",
            data=vcsv,
            file_name="meridional_velocity.csv",
            mime="text/csf",
        )
        st.download_button(
            label="Download Vertical Velocity File (CSV)",
            data=vcsv,
            file_name="vertical_velocity.csv",
            mime="text/csf",
        )

        st.download_button(
        label="Download Final Mask (CSV)",
        data=csv_mask,
        file_name="final_mask.csv",
        mime="text/csv",
        )


# Option to Download Config file
# ------------------------------

# Header for the Config.ini File Generator
st.header("Config.ini File Generator", divider="blue")

# Radio button to decide whether to generate the config.ini file
generate_config_radio = st.radio(
    "Do you want to generate a config.ini file?", ("No", "Yes")
)


if generate_config_radio == "Yes":
    # Create a config parser object
    config = configparser.ConfigParser()

    # Main section
    config["FileSettings"] = {}
    config["DownloadOptions"] = {}
    config["QCTest"] = {"qc_test": "False"}
    config["ProfileTest"] = {"profile_test": "False"}
    config["VelocityTest"] = {"velocity_test": "False"}
    config["Optional"] = {"attributes": "False"}

    config["FileSettings"]["input_file_path"] = ""
    config["FileSettings"]["input_file_name"] = st.session_state.fname
    config["FileSettings"]["output_file_path"] = ""
    config["FileSettings"]["output_file_name_raw"] = ""
    config["FileSettings"]["output_file_name_processed"] = ""
    config["FileSettings"]["output_format_raw"] = str(file_type_radio).lower()
    config["FileSettings"]["output_format_processed"] = str(file_type_radio).lower()

    config["DownloadOptions"]["download_raw"] = "True"
    config["DownloadOptions"]["download_processed"] = "True"
    config["DownloadOptions"]["apply_mask"] = "True"
    config["DownloadOptions"]["download_mask"] = "True"

    # QC Test Options
    if st.session_state.isQCMask:
        config["QCTest"]["qc_test"] = "True"

        # Add the contents of the current QC Mask thresholds
        if "newthresh" in st.session_state:
            for idx, row in st.session_state.newthresh.iterrows():
                config["QCTest"][row["Threshold"].replace(" ", "_")] = row["Values"]

    # Profile Test Options
    if st.session_state.isProfileMask:
        config["ProfileTest"]["profile_test"] = "True"

        if st.session_state.isTrimEnds:
            config["ProfileTest"]["trim_ends"] = "True"
            config["ProfileTest"]["trim_ends_start_index"] = str(
                st.session_state.start_ens
            )
            config["ProfileTest"]["trim_ends_end_index"] = str(st.session_state.end_ens)
        else:
            config["ProfileTest"]["trim_ends"] = "False"

        if st.session_state.isCutBins:
            config["ProfileTest"]["cut_bins"] = "True"
            config["ProfileTest"]["cut_bins_add_cells"] = str(
                st.session_state.extra_cells
            )
        else:
            config["ProfileTest"]["cut_bins"] = "False"

        if st.session_state.isGrid:
            config["ProfileTest"]["regrid"] = "True"
            config["ProfileTest"][
                "Regrid_Option"
            ] = st.session_state.end_bin_option
        else:
            config["ProfileTest"]["regrid"] = "False"

    # Velocity Test Section
    if st.session_state.isVelocityMask:
        config["VelocityTest"]["velocity_test"] = "True"

        if st.session_state.isMagnet:
            config["VelocityTest"]["magnetic_declination"] = str(True)
            config["VelocityTest"]["latitude"] = str(st.session_state.lat)
            config["VelocityTest"]["longitude"] = str(st.session_state.lon)
            config["VelocityTest"]["depth"] = str(st.session_state.magnetic_dec_depth)
            config["VelocityTest"]["year"] = str(st.session_state.year)
        else:
            config["VelocityTest"]["magnetic_declination"] = str(False)

        if st.session_state.isCutoff:
            config["VelocityTest"]["cutoff"] = str(True)
            config["VelocityTest"]["max_zonal_velocity"] = str(st.session_state.maxuvel)
            config["VelocityTest"]["max_meridional_velocity"] = str(
                st.session_state.maxvvel
            )
            config["VelocityTest"]["max_vertical_velocity"] = str(
                st.session_state.maxwvel
            )
        else:
            config["VelocityTest"]["cutoff"] = str(False)

        if st.session_state.isDespike:
            config["VelocityTest"]["despike"] = str(True)
            config["VelocityTest"]["despike_Kernal_Size"] = str(
                st.session_state.despike_kernal
            )
            config["VelocityTest"]["despike_Cutoff"] = str(
                st.session_state.despike_cutoff
            )
        else:
            config["VelocityTest"]["Despike"] = str(False)

        if st.session_state.isFlatline:
            config["VelocityTest"]["flatline"] = str(True)
            config["VelocityTest"]["flatline_kernal_size"] = str(
                st.session_state.flatline_kernal
            )
            config["VelocityTest"]["flatline_deviation"] = str(
                st.session_state.flatline_cutoff
            )
        else:
            config["VelocityTest"]["flatline"] = str(False)

    # Optional section (attributes)
    config["Optional"] = {}
    for key, value in st.session_state.attributes.items():
        config["Optional"][key] = str(value)  # Ensure all values are strings

    # Write config.ini to a temporary file
    config_filepath = "config.ini"
    with open(config_filepath, "w") as configfile:
        config.write(configfile)

    # Allow the user to download the generated config.ini file
    with open(config_filepath, "rb") as file:
        st.download_button(
            label="Download config.ini File",
            data=file,
            file_name="config.ini",
        )

    display_config_radio = st.radio(
        "Do you want to display config.ini file?", ("No", "Yes")
    )
    if display_config_radio == "Yes":
        st.write({section: dict(config[section]) for section in config.sections()})
