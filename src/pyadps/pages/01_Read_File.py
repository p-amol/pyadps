import os
import tempfile

import pandas as pd
import streamlit as st
import utils.readrdi as rd
import utils.writenc as wr
from streamlit.runtime.state import session_state
from utils.signal_quality import default_mask

"""
Streamlit page to load ADCP binary file and display File Header
and Fixed Leader data
"""

if "fname" not in st.session_state:
    st.session_state.fname = "No file selected"

if "rawfilename" not in st.session_state:
    st.session_state.rawfilename = "rawfile.nc"

if "vleadfilename" not in st.session_state:
    st.session_state.vleadfilename = "vlead.nc"


################ Functions #######################
@st.cache_data()
def file_access(uploaded_file):
    """
    Function creates temporary directory to store the uploaded file.
    The path of the file is returned

    Args:
        uploaded_file (string): Name of the uploaded file

    Returns:
        path (string): Path of the uploaded file
    """
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def color_bool(val):
    """
    Takes a scalar and returns a string with
    the css color property.
    """
    if isinstance(val, bool):
        if val:
            color = "green"
        else:
            color = "red"
    else:
        color = "orange"
    return "color: %s" % color


def color_bool2(val):
    """
    Takes a scalar and returns a string with
    the css color property. The following colors
    are assinged for the string
        "True": green,
        "False": red
         Any other string: orange

    Args:
        val (string): Any string data

    Returns:
        The input string with css color property added
    """
    if val == "True" or val == "Data type is healthy":
        color = "green"
    elif val == "False":
        color = "red"
    # elif val in st.session_state.ds.warnings.values():
    #     color = "orange"
    else:
        color = "orange"
    return "color: %s" % color


@st.cache_data
def read_file(filepath):
    ds = rd.ReadFile(st.session_state.fpath)
    if not ds.isEnsembleEqual:
        ds.fixensemble()
    st.session_state.ds = ds
    # return ds


uploaded_file = st.file_uploader("Upload RDI ADCP Binary File", type="000")

if uploaded_file is not None:
    # st.cache_data.clear

    # Get path
    st.session_state.fpath = file_access(uploaded_file)
    # Get data
    read_file(st.session_state.fpath)
    ds = st.session_state.ds
    head = ds.fileheader
    flead = ds.fixedleader
    vlead = ds.variableleader
    velocity = ds.velocity.data
    correlation = ds.correlation.data
    echo = ds.echo.data
    pgood = ds.percentgood.data

    st.session_state.fname = uploaded_file.name
    st.session_state.head = ds.fileheader
    st.session_state.flead = ds.fixedleader
    st.session_state.vlead = ds.variableleader
    st.session_state.velocity = ds.velocity.data
    st.session_state.echo = ds.echo.data
    st.session_state.correlation = ds.correlation.data
    st.session_state.pgood = ds.percentgood.data

    # st.session_state.flead = flead
    # st.session_state.vlead = vlead
    # st.session_state.head = head
    # st.session_state.velocity = velocity
    # st.session_state.echo = echo
    # st.session_state.correlation = correlation
    # st.session_state.pgood = pgood
    st.write("You selected `%s`" % st.session_state.fname)

elif "flead" in st.session_state:
    st.write("You selected `%s`" % st.session_state.fname)
else:
    st.stop()

########## TIME AXIS ##############

# Time axis is extracted and stored as Pandas datetime
year = st.session_state.vlead.vleader["RTC Year"]
month = st.session_state.vlead.vleader["RTC Month"]
day = st.session_state.vlead.vleader["RTC Day"]
hour = st.session_state.vlead.vleader["RTC Hour"]
minute = st.session_state.vlead.vleader["RTC Minute"]
second = st.session_state.vlead.vleader["RTC Second"]

# Recent ADCP binary files have Y2K compliant clock. The Century
# is stored in`RTC Century`. As all files may not have this clock
# we have added 2000 to the year.
# CHECKS:
# Are all our data Y2K compliant?
# Should we give users the options to correct the data?

year = year + 2000
date_df = pd.DataFrame(
    {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
    }
)

st.session_state.date = pd.to_datetime(date_df)

######### MASK DATA ##############
# The velocity data has missing values due to the cutoff
# criteria used before deployment. The `default_mask` uses
# the velocity to create a mask. This mask  file is stored
# in the session_state.
#
# WARNING: Never Change `st.session_state.orig_mask` in the code!
#
if "orig_mask" not in st.session_state:
    st.session_state.orig_mask = default_mask(
        st.session_state.flead, st.session_state.velocity
    )

# Checks if the following quality checks are carried out
st.session_state.isQCMask = False
st.session_state.isProfileMask = False
st.session_state.isGrid = False
st.session_state.isGridSave = False
st.session_state.isVelocityMask = False

########## FILE HEADER ###############
st.header("File Header", divider="blue")
st.write(
    """
        Header information is the first item sent by the ADCP. You may check the file size, total ensembles, and available data types. The function also checks if the total bytes and data types are uniform for all ensembles. 
        """
)

left1, right1 = st.columns(2)
with left1:
    check_button = st.button("Check File Health")
    if check_button:
        cf = st.session_state.head.check_file()
        if (
            cf["File Size Match"]
            and cf["Byte Uniformity"]
            and cf["Data Type Uniformity"]
        ):
            st.write("Your file appears healthy! :sunglasses:")
        else:
            st.write("Your file appears corrupted! :worried:")

        cf["File Size (MB)"] = "{:,.2f}".format(cf["File Size (MB)"])
        st.write(f"Total no. of Ensembles: :green[{st.session_state.head.ensembles}]")
        df = pd.DataFrame(cf.items(), columns=pd.array(["Check", "Details"]))
        df = df.astype("str")
        st.write(df.style.map(color_bool2, subset="Details"))
        # st.write(df)
with right1:
    datatype_button = st.button("Display Data Types")
    if datatype_button:
        st.write(
            pd.DataFrame(
                st.session_state.head.data_types(),
                columns=pd.array(["Available Data Types"]),
            )
        )

if st.session_state.ds.isWarning:
    st.write(
        """ 
            Warnings detected while reading. Data sets may still be available for processing.
            Click `Display Warning` to display warnings for each data types.
        """
    )
    warning_button = st.button("Display Warnings")
    df2 = pd.DataFrame(
        st.session_state.ds.warnings.items(),
        columns=pd.array(["Data Type", "Warnings"]),
    )
    if warning_button:
        st.write(df2.style.map(color_bool2, subset=["Warnings"]))

############ FIXED LEADER #############

st.header("Fixed Leader (Static Variables)", divider="blue")
st.write(
    """
        Fixed Leader data refers to the non-dynamic WorkHorse ADCP data like the hardware information and the thresholds. Typically, values remain constant over time. They only change when you change certain commands, although there are occasional exceptions. You can confirm this using the :blue[**Fleader Uniformity Check**]. Click :blue[**Fixed Leader**] to display the values for the first ensemble.
        """
)


flead_check_button = st.button("Fleader Uniformity Check")
if flead_check_button:
    st.write("The following variables are non-uniform:")
    for keys, values in st.session_state.flead.is_uniform().items():
        if not values:
            st.markdown(f":blue[**- {keys}**]")
    st.write("Displaying all static variables")
    df = pd.DataFrame(st.session_state.flead.is_uniform(), index=[0]).T
    st.write(df.style.map(color_bool))

flead_button = st.button("Fixed Leader")
if flead_button:
    df = pd.DataFrame(
        {
            "Fields": st.session_state.flead.field().keys(),
            "Values": st.session_state.flead.field().values(),
        }
    )
    st.dataframe(df, use_container_width=True)

left, centre, right = st.columns(3)
with left:
    st.dataframe(st.session_state.flead.system_configuration())

with centre:
    st.dataframe(st.session_state.flead.ez_sensor())
    #     st.write(output)
with right:
    # st.write(st.session_state.flead.ex_coord_trans())
    df = pd.DataFrame(st.session_state.flead.ex_coord_trans(), index=[0]).T
    df = df.astype("str")
    st.write((df.style.map(color_bool2)))
    # st.dataframe(df)

