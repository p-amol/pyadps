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
    st.session_state.isAttributes = False

if st.session_state.isVelocityTest:
    st.session_state.final_mask = st.session_state.velocity_mask

    if st.session_state.isVelocityModifiedMagnet:
        st.session_state.final_velocity = st.session_state.velocity_magnet
    if st.session_state.isRegridCheck_PT:
        st.session_state.final_velocity = st.session_state.velocity_regrid
    elif st.session_state.isVelocityModifiedSound_ST:
        st.session_state.final_velocity = st.session_state.velocity_sensor
    else:
        st.session_state.final_velocity = st.session_state.velocity

    if st.session_state.isRegridCheck_PT:
        st.session_state.final_echo = st.session_state.echo_regrid
        st.session_state.final_correlation = st.session_state.correlation_regrid
        st.session_state.final_pgood = st.session_state.pgood_regrid
    else:
        st.session_state.final_echo = st.session_state.echo
        st.session_state.final_correlation = st.session_state.correlation
        st.session_state.final_pgood = st.session_state.pgood
else:
    if st.session_state.isRegridCheck_PT:
        st.session_state.final_mask = st.session_state.profile_mask_regrid
        st.session_state.final_velocity = st.session_state.velocity_regrid
        st.session_state.final_echo = st.session_state.echo_regrid
        st.session_state.final_correlation = st.session_state.correlation_regrid
        st.session_state.final_pgood = st.session_state.pgood_regrid
    else:
        if st.session_state.isProfileTest:
            st.session_state.final_mask = st.session_state.profile_mask
        elif st.session_state.isQCTest:
            st.session_state.final_mask = st.session_state.qc_mask
        elif st.session_state.isSensorTest:
            st.session_state.final_mask = st.session_state.sensor_mask
        else:
            st.session_state.final_mask = st.session_state.orig_mask
        st.session_state.final_velocity = st.session_state.velocity
        st.session_state.final_echo = st.session_state.echo
        st.session_state.final_correlation = st.session_state.correlation
        st.session_state.final_pgood = st.session_state.pgood


if "depth_axis" not in st.session_state:
    st.session_state.isRegridCheck_PT = False


@st.cache_data
def file_write(filename="processed_file.nc"):
    tempdirname = tempfile.TemporaryDirectory(delete=False)
    outfilepath = tempdirname.name + "/" + filename
    return outfilepath


# If the data is not regrided based on pressure sensor. Use the mean depth
if not st.session_state.isRegridCheck_PT:
    st.write(":red[WARNING!]")
    st.write(
        "Data not regrided. Using the mean transducer depth to calculate the depth axis."
    )
    # mean_depth = np.mean(st.session_state.vlead.vleader["Depth of Transducer"]) / 10
    mean_depth = np.mean(st.session_state.depth) / 10
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

st.session_state.mask_data_WF = st.radio(
    "Do you want to mask the final data?", ("Yes", "No")
)

if st.session_state.mask_data_WF == "Yes":
    mask = st.session_state.final_mask
    st.session_state.write_velocity = np.copy(st.session_state.final_velocity)
    st.session_state.write_velocity[:, mask == 1] = -32768
else:
    st.session_state.write_velocity = np.copy(st.session_state.final_velocity)


st.session_state.file_type_WF = st.radio(
    "Select output file format:", ("NetCDF", "CSV")
)

if st.session_state.file_type_WF == "NetCDF":
    add_attr_button = st.checkbox("Add attributes to NetCDF file")

    if add_attr_button:
        st.session_state.isAttributes = True
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

    if st.session_state.file_type_WF == "NetCDF":
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

    if st.session_state.file_type_WF == "CSV":
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
    config["SensorTest"] = {"sensor_test": "False"}
    config["QCTest"] = {"qc_test": "False"}
    config["ProfileTest"] = {"profile_test": "False"}
    config["VelocityTest"] = {"velocity_test": "False"}
    config["Attributes"] = {}

    # ------------------
    # File Settings
    # ------------------
    config["FileSettings"]["input_file_path"] = ""
    config["FileSettings"]["input_file_name"] = st.session_state.fname
    config["FileSettings"]["output_file_path"] = ""
    config["FileSettings"]["output_file_name_raw_netcdf"] = ""
    config["FileSettings"]["output_file_name_vlead_netcdf"] = ""
    config["FileSettings"]["output_file_name_raw_csv"] = ""
    config["FileSettings"]["output_file_name_processed_netcdf"] = ""
    config["FileSettings"]["output_file_name_processed_csv"] = ""

    if st.session_state.file_type_WF.lower() == "netcdf":
        st.session_state.isProcessedNetcdfDownload_WF = True
    else:
        st.session_state.isProcessedNetcdfDownload_WF = False
        st.session_state.isProcessedCSVDownload_WF = True
    # ------------------
    # Download options
    # ------------------
    config["DownloadOptions"]["download_raw_netcdf"] = str(
        st.session_state.rawnc_download_DRW
    )
    config["DownloadOptions"]["download_vlead_netcdf"] = str(
        st.session_state.vleadnc_download_DRW
    )
    config["DownloadOptions"]["download_processed_netcdf"] = str(
        st.session_state.isProcessedNetcdfDownload_WF
    )
    config["DownloadOptions"]["download_raw_csv"] = str(
        st.session_state.rawcsv_download_DRW
    )
    config["DownloadOptions"]["download_processed_csv"] = str(
        st.session_state.isProcessedCSVDownload_WF
    )
    config["DownloadOptions"]["add_attributes_raw"] = str(
        st.session_state.add_attributes_DRW
    )
    config["DownloadOptions"]["add_attributes_processed"] = str(
        st.session_state.isAttributes
    )
    config["DownloadOptions"]["axis_option"] = str(st.session_state.axis_option_DRW)
    config["DownloadOptions"]["apply_mask"] = "True"
    config["DownloadOptions"]["download_mask"] = "True"

    # ------------------
    # PAGE: Sensor Test
    # ------------------
    config["SensorTest"]["sensor_test"] = str(st.session_state.isSensorTest)
    # Tab 1: Depth Sensor
    config["SensorTest"]["is_depth_modified"] = str(st.session_state.isDepthModified_ST)
    config["SensorTest"]["depth_input_option"] = str(st.session_state.depthoption_ST)
    config["SensorTest"]["is_fixed_depth"] = str(st.session_state.isFixedDepth_ST)
    config["SensorTest"]["fixed_depth"] = str(st.session_state.fixeddepth_ST)
    config["SensorTest"]["is_upload_depth"] = str(st.session_state.isUploadDepth_ST)
    config["SensorTest"]["depth_file_path"] = ""

    # Tab 2: Salinity sensor
    config["SensorTest"]["is_salinity_modified"] = str(
        st.session_state.isSalinityModified_ST
    )
    config["SensorTest"]["salinity_input_option"] = str(
        st.session_state.salinityoption_ST
    )
    config["SensorTest"]["is_fixed_salinity"] = str(st.session_state.isFixedSalinity_ST)
    config["SensorTest"]["fixed_salinity"] = str(st.session_state.fixedsalinity_ST)
    config["SensorTest"]["is_upload_salinity"] = str(
        st.session_state.isUploadSalinity_ST
    )
    config["SensorTest"]["salinity_file_path"] = ""

    # Tab 3: Temperature sensor
    config["SensorTest"]["is_temperature_modified"] = str(
        st.session_state.isTemperatureModified_ST
    )
    config["SensorTest"]["temperature_input_option"] = str(
        st.session_state.temperatureoption_ST
    )
    config["SensorTest"]["is_fixed_temperature"] = str(
        st.session_state.isFixedTemperature_ST
    )
    config["SensorTest"]["fixed_temperature"] = str(
        st.session_state.fixedtemperature_ST
    )
    config["SensorTest"]["is_upload_temperature"] = str(
        st.session_state.isUploadTemperature_ST
    )
    config["SensorTest"]["temperature_file_path"] = ""

    # Tab 7:

    config["SensorTest"]["roll_check"] = str(st.session_state.isRollCheck_ST)
    config["SensorTest"]["roll_cutoff"] = str(st.session_state.roll_cutoff_ST)
    config["SensorTest"]["pitch_check"] = str(st.session_state.isPitchCheck_ST)
    config["SensorTest"]["pitch_cutoff"] = str(st.session_state.pitch_cutoff_ST)

    config["SensorTest"]["velocity_modified"] = str(
        st.session_state.isVelocityModifiedSound_ST
    )

    # ------------------
    # PAGE: QC Test
    # ------------------
    # Tab 2
    config["QCTest"]["qc_test"] = str(st.session_state.isQCTest)
    config["QCTest"]["qc_check"] = str(st.session_state.isQCCheck_QCT)
    config["QCTest"]["correlation"] = str(st.session_state.ct_QCT)
    config["QCTest"]["echo_intensity"] = str(st.session_state.et_QCT)
    config["QCTest"]["error_velocity"] = str(st.session_state.evt_QCT)
    config["QCTest"]["false_target"] = str(st.session_state.ft_QCT)
    config["QCTest"]["three_beam"] = str(st.session_state.is3beam_QCT)
    config["QCTest"]["percent_good"] = str(st.session_state.pgt_QCT)

    # Tab 4
    config["QCTest"]["beam_modified"] = str(st.session_state.isBeamModified_QCT)
    config["QCTest"]["orientation"] = str(st.session_state.beam_direction_QCT)

    # ------------------
    # PAGE: Profile Test
    # ------------------
    # Tab 1
    config["ProfileTest"]["profile_test"] = str(st.session_state.isProfileTest)
    config["ProfileTest"]["trim_ends_check"] = str(st.session_state.isTrimEndsCheck_PT)
    config["ProfileTest"]["trim_start_ensemble"] = str(st.session_state.start_ens_PT)
    config["ProfileTest"]["trim_end_ensemble"] = str(st.session_state.end_ens_PT)

    # Tab 2
    config["ProfileTest"]["cutbins_sidelobe_check"] = str(
        st.session_state.isCutBinSideLobeCheck_PT
    )
    config["ProfileTest"]["extra_cells"] = str(st.session_state.extra_cells_PT)
    config["ProfileTest"]["water_depth"] = str(st.session_state.water_depth_PT)

    # Tab 3
    config["ProfileTest"]["manual_cutbins"] = str(
        st.session_state.isCutBinManualCheck_PT
    )

    # Tab 4
    config["ProfileTest"]["regrid"] = str(st.session_state.isRegridCheck_PT)
    config["ProfileTest"]["end_cell_option"] = str(st.session_state.end_cell_option_PT)
    config["ProfileTest"]["interpolate"] = str(st.session_state.interpolate_PT)
    config["ProfileTest"]["boundary"] = str(st.session_state.manualdepth_PT)

    # ------------------
    # PAGE: Velocity Test
    # ------------------

    config["VelocityTest"]["velocity_test"] = str(st.session_state.isVelocityTest)

    # Tab 1
    config["VelocityTest"]["magnetic_declination"] = str(
        st.session_state.isMagnetCheck_VT
    )
    config["VelocityTest"]["magnet_method"] = str(st.session_state.magnet_method_VT)
    config["VelocityTest"]["magnet_latitude"] = str(st.session_state.magnet_lat_VT)
    config["VelocityTest"]["magnet_longitude"] = str(st.session_state.magnet_lon_VT)
    config["VelocityTest"]["magnet_depth"] = str(st.session_state.magnet_depth_VT)
    config["VelocityTest"]["magnet_year"] = str(st.session_state.magnet_year_VT)
    config["VelocityTest"]["magnet_user_input"] = str(
        st.session_state.magnet_user_input_VT
    )

    # Tab 2
    config["VelocityTest"]["cutoff"] = str(st.session_state.isCutoffCheck_VT)
    config["VelocityTest"]["max_zonal_velocity"] = str(st.session_state.maxuvel_VT)
    config["VelocityTest"]["max_meridional_velocity"] = str(st.session_state.maxvvel_VT)
    config["VelocityTest"]["max_vertical_velocity"] = str(st.session_state.maxwvel_VT)

    # Tab 3
    config["VelocityTest"]["despike"] = str(st.session_state.isDespikeCheck_VT)
    config["VelocityTest"]["despike_kernal_size"] = str(
        st.session_state.despike_kernal_VT
    )
    config["VelocityTest"]["despike_cutoff"] = str(st.session_state.despike_cutoff_VT)

    # Tab 4
    config["VelocityTest"]["flatline"] = str(st.session_state.isFlatlineCheck_VT)
    config["VelocityTest"]["flatline_kernal_size"] = str(
        st.session_state.flatline_kernal_VT
    )
    config["VelocityTest"]["flatline_cutoff"] = str(st.session_state.flatline_cutoff_VT)

    # Optional section (attributes)

    for key, value in st.session_state.attributes.items():
        config["Attributes"][key] = str(value)  # Ensure all values are strings

    # Write config.ini to a temporary file
    # config_filepath = "config.ini"
    # with open(config_filepath, "w") as configfile:
    #     config.write(configfile)
    # Create a temporary file for the config.ini
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".ini") as temp_config:
        config.write(temp_config)
        temp_config_path = temp_config.name
    # Allow the user to download the generated config.ini file
    with open(temp_config_path, "rb") as file:
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
