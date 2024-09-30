import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly_resampler import FigureResampler

if "flead" not in st.session_state:
    st.write(":red[Please Select Data!]")
    st.stop()

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


@st.cache_data
def fillplot_matplotlib(data):
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, data)
    fig.colorbar(cs)
    st.pyplot(fig)


@st.cache_data
def fillplot_plotly(data, colorscale="balance", title="Data", xaxis="time"):
    if xaxis == "time":
        xdata = st.session_state.date
    elif xaxis == "ensemble":
        xdata = x
    else:
        xdata = x
    fig = FigureResampler(go.Figure())
    data1 = np.where(data == -32768, np.nan, data)
    fig.add_trace(
        go.Heatmap(
            z=data1[:, 0:-1],
            x=xdata,
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


@st.cache_data
def lineplot(data, title, xaxis="time"):
    if xaxis == "time":
        df = pd.DataFrame({"date": st.session_state.date, title: data})
        fig = px.line(df, x="date", y=title)
    else:
        df = pd.DataFrame({"ensemble": x, title: data})
        fig = px.line(df, x="ensemble", y=title)

    st.plotly_chart(fig)


# Introduction
st.header("View Raw Data", divider="orange")
st.write(
    """
Displays all variables available in the raw file. No processing has been carried out. 
Data might be missing because of the quality-check criteria used before deployment.\n 
Either `time` or `ensemble` axis can be chosen as the abscissa (x-axis).
The ordinate (y-axis) for the heatmap is `bins` as the depth correction is not applied. 
"""
)
xbutton = st.radio("Select an x-axis to plot", ["time", "ensemble"], horizontal=True)


# Fixed Leader Plots
st.header("Fixed Leader", divider="blue")
fbutton = st.radio("Select a dynamic variable to plot:", fdata.keys(), horizontal=True)
lineplot(fdata[fbutton], fbutton, xaxis=str(xbutton))

# Variable Leader Plots
st.header("Variable Leader", divider="blue")
vbutton = st.radio("Select a dynamic variable to plot:", vdata.keys(), horizontal=True)
lineplot(vdata[vbutton], vbutton, xaxis=str(xbutton))

basic_options = [
    "Pressure",
    "Temperature",
    "Salinity",
    "Depth of Transducer",
    "Heading",
    "Pitch",
    "Roll",
]


st.header("Velocity, Echo Intensity, Correlation & Percent Good", divider="blue")


def call_plot(varname, beam, xaxis="time"):
    if varname == "Velocity":
        fillplot_plotly(velocity[beam - 1, :, :], title=varname, xaxis=xaxis)
    elif varname == "Echo":
        fillplot_plotly(echo[beam - 1, :, :], title=varname, xaxis=xaxis)
    elif varname == "Correlation":
        fillplot_plotly(correlation[beam - 1, :, :], title=varname, xaxis=xaxis)
    elif varname == "Percent Good":
        fillplot_plotly(pgood[beam - 1, :, :], title=varname, xaxis=xaxis)


var_option = st.selectbox(
    "Select a data type", ("Velocity", "Echo", "Correlation", "Percent Good")
)
beam = st.radio("Select beam", (1, 2, 3, 4), horizontal=True)
call_plot(var_option, beam, xaxis=str(xbutton))


