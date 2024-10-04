import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit.runtime.state import session_state
import utils.writenc as wr

# Session states initialization
if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()

if "fname" not in st.session_state:
    st.session_state.fname = "No file selected"

if "rawfilename" not in st.session_state:
    st.session_state.rawfilename = "rawfile.nc"

if "vleadfilename" not in st.session_state:
    st.session_state.vleadfilename = "vlead.nc"

if "attributes" not in st.session_state:
    st.session_state.attributes = {}
    
if "add_attributes" not in st.session_state:
    st.session_state.add_attributes = "No"  # Default value

################ Functions #######################
@st.cache_data()
def file_access(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path

@st.cache_data
def read_file(filepath):
    ds = rd.ReadFile(st.session_state.fpath)
    if not ds.isEnsembleEqual:
        ds.fixensemble()
    st.session_state.ds = ds

@st.cache_data
def file_write(path, add_attributes=True):
    tempdirname = tempfile.TemporaryDirectory(delete=False)
    st.session_state.rawfilename = tempdirname.name + "/rawfile.nc"
    
    if add_attributes:
        wr.main(path, st.session_state.rawfilename, attributes=st.session_state.attributes)
    else:
        wr.main(path, st.session_state.rawfilename)

@st.cache_data
def file_write_vlead(path, add_attributes=True):
    tempvardirname = tempfile.TemporaryDirectory(delete=False)
    st.session_state.vleadfilename = tempvardirname.name + "/vlead.nc"
    
    if add_attributes:
        wr.vlead_nc(path, st.session_state.vleadfilename, attributes=st.session_state.attributes)
    else:
        wr.vlead_nc(path, st.session_state.vleadfilename)

# UI for attribute selection
st.header("NetCDF File", divider="blue")

# Option to add attributes
add_attributes = st.radio("Do you want to add attributes to the NetCDF file?", ["No", "Yes"], horizontal=True)

if add_attributes == "Yes":
    st.write("### Please fill in the attributes:")
    
    # Two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.attributes['Cruise_No.'] = st.text_input("Cruise No.")
        st.session_state.attributes['Ship_Name'] = st.text_input("Ship Name")
        st.session_state.attributes['Project_No.'] = st.text_input("Project No.")
        st.session_state.attributes['Water_Depth_m'] = st.text_input("Water Depth (m)")
        st.session_state.attributes['Deployment_Depth_m'] = st.text_input("Deployment Depth (m)")
        st.session_state.attributes['Deployment_Date'] = st.date_input("Deployment Date")
        st.session_state.attributes['Recovery_Date'] = st.date_input("Recovery Date")


    with col2:
        st.session_state.attributes['Latitude'] = st.text_input("Latitude")
        st.session_state.attributes['Longitude'] = st.text_input("Longitude")
        st.session_state.attributes['Platform_Type'] = st.text_input("Platform Type")
        st.session_state.attributes['Participants'] = st.text_area("Participants")
        st.session_state.attributes['File_created_by'] = st.text_input("File created by")
        st.session_state.attributes['Contact'] = st.text_input("Contact")
        st.session_state.attributes['Comments'] = st.text_area("Comments")

    st.write("Attributes will be added to the NetCDF file once you submit.")
    
# Buttons to generate files
download_button = st.button("Generate Raw NetCDF File")
download_var_button = st.button("Generate Raw Variable Leader NetCDF File")

if download_button:
    file_write(st.session_state.fpath, add_attributes == "Yes")
    st.write(st.session_state.rawfilename)
    with open(st.session_state.rawfilename, "rb") as file:
        st.download_button(
            label="Download Raw File",
            data=file,
            file_name="rawfile.nc",
        )

if download_var_button:
    file_write_vlead(st.session_state.fpath, add_attributes == "Yes")
    st.write(st.session_state.vleadfilename)
    with open(st.session_state.vleadfilename, "rb") as file:
        st.download_button(
            label="Download Variable Leader",
            data=file,
            file_name="vlead.nc",
        )

def download_csv_with_ensemble(data, filename):
    # Create ensemble numbers from 1 to the number of rows in the data
    ensembles = np.arange(1, len(next(iter(data.values()))) + 1)

    # Convert data to a DataFrame and insert ensembles as the first column
    df = pd.DataFrame(data)
    df.insert(0, "RDI_Ensemble", ensembles)  # Add ensemble numbers as the first column

    # Export the DataFrame as a CSV
    csv = df.to_csv(index=False).encode('utf-8')
    return st.download_button(
        label=f"Download {filename} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime='text/csv',
    )
    
def download_csv(data, filename):
    # Convert data to DataFrame if it's not already one
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient="index").T
    else:
        df = pd.DataFrame(data)
    
    # Export the DataFrame as a CSV
    csv = df.to_csv(index=False).encode('utf-8')
    return st.download_button(
        label=f"Download {filename} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime='text/csv',
    )
    
    
def download_csv1(data, filename):
    # Convert data to DataFrame if it's not already one
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient="index").T
    else:
        df = pd.DataFrame(data)
    
    # Create ensemble and depth arrays
    ensembles = np.arange(1, df.shape[0] + 1)
    depths = np.arange(1, df.shape[1] + 1)
    
    # Add ensemble numbers as the first column
    df.insert(0, 'Ensemble', ensembles)
    
    # Transpose the DataFrame to switch rows and columns
    df = df.T
    
    # Add depth values as the first row
    df.insert(0, 'Depth', [''] + list(depths))
    
    # Export the DataFrame as a CSV
    csv = df.to_csv(index=False, header=False).encode('utf-8')
    return st.download_button(
        label=f"Download {filename} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime='text/csv',
    )

# Load data
fdata = st.session_state.flead.fleader
vdata = st.session_state.vlead.vleader
velocity = st.session_state.velocity
echo = st.session_state.echo
correlation = st.session_state.correlation
pgood = st.session_state.pgood

x = np.arange(0, st.session_state.head.ensembles, 1)
y = np.arange(0, fdata["Cells"][0], 1)

X, Y = np.meshgrid(x, y)

# Unified download section
st.header("Download Raw Data CSV File", divider="blue")

# Selection for the data category
data_type = st.selectbox(
    "Select data type to download:",
    ["Fixed Leader", "Variable Leader", "Velocity", "Echo Intensity", "Correlation", "Percent Good"]
)

# Show corresponding variable options based on selection
if data_type == "Fixed Leader":
    # Combine all variables of Fixed Leader into one DataFrame
    f_combined_data = {var: fdata[var] for var in fdata.keys()}
    download_csv_with_ensemble(f_combined_data, "Fixed_Leader_All_Variables")

elif data_type == "Variable Leader":
    # Combine all variables of Variable Leader into one DataFrame
    v_combined_data = {var: vdata[var] for var in vdata.keys()}
    download_csv(v_combined_data, "Variable_Leader_All_Variables")

else:
    beam_download = st.radio("Select beam to download", (1, 2, 3, 4), horizontal=True)

    if data_type == "Velocity":
        download_data = velocity[beam_download - 1, :, :]
    elif data_type == "Echo Intensity":
        download_data = echo[beam_download - 1, :, :]
    elif data_type == "Correlation":
        download_data = correlation[beam_download - 1, :, :]
    elif data_type == "Percent Good":
        download_data = pgood[beam_download - 1, :, :]

    download_csv1(download_data, f"{data_type}_Beam_{beam_download}")

