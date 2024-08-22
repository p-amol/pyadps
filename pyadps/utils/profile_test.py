import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.widgets import Button, Slider, TextBox

from pyadps.utils import readrdi as rd
from pyadps.utils.plotgen import plotmask, plotvar


class PlotEnds:
    """
    A class to interactively select the start and end ensembles for trimming ADCP data.

    Attributes:
        pressure (numpy.ndarray): Array of pressure values.
        delta (int): Interval for selecting ensemble limits.
        n (int): Total number of ensembles.
        start_ens (int): Selected starting ensemble.
        end_ens (int): Selected ending ensemble.

    Methods:
        update1(value): Updates the plot for start ensemble selection.
        update2(value): Updates the plot for end ensemble selection.
        show(): Displays the interactive plot window.
        exitwin(event): Closes the plot window and saves the selections.
    """
    def __init__(self, pressure, delta=10):
        """
        Initializes the PlotEnds class with pressure data and delta for ensemble selection.

        Args:
            pressure (numpy.ndarray): Array of pressure values. delta (int, optional): Interval for selecting ensemble limits. Default is 10.
        """
        self.dep = pressure / 980

        self.n = np.size(self.dep)
        self.delta = delta
        self.nmin = 0
        self.nmax = self.nmin + self.delta
        self.mmax = 0
        self.mmin = self.mmax - self.delta

        self.x = np.arange(0, self.n)

        self.start_ens = 0
        self.end_ens = 0

        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 8))
        self.fig.set_facecolor("darkgrey")
        plt.subplots_adjust(bottom=0.28, right=0.72)

        self.ax_end = self.fig.add_axes(rect=(0.25, 0.08, 0.47, 0.03))
        self.ax_start = self.fig.add_axes(rect=(0.25, 0.15, 0.47, 0.03))
        self.ax_button = self.fig.add_axes(rect=(0.81, 0.05, 0.15, 0.075))
        # self.ax_depmaxbutton = self.fig.add_axes(rect=(0.68, 0.13, 0.04, 0.02))
        # self.ax_depminbutton = self.fig.add_axes(rect=(0.25, 0.13, 0.04, 0.02))
        # self.ax_recmaxbutton = self.fig.add_axes(rect=(0.68, 0.06, 0.04, 0.02))
        # self.ax_recminbutton = self.fig.add_axes(rect=(0.25, 0.06, 0.04, 0.02))

        # Plot
        self.axs[0].scatter(self.x, self.dep, color="k")
        self.axs[1].scatter(self.x, self.dep, color="k")

        # Figure Labels
        for i in range(2):
            self.axs[i].set_xlabel("Ensemble")
        self.axs[0].set_xlim([self.nmin - 1, self.nmax])
        self.axs[1].set_xlim([self.n - self.delta, self.n])
        self.axs[0].set_ylabel("Depth (m)")
        self.fig.suptitle("Trim Ends")

        # Display statistics
        self.axs[0].text(0.82, 0.60, "Statistics", transform=plt.gcf().transFigure)
        self.max = np.round(np.max(self.dep), decimals=2)
        self.min = np.round(np.min(self.dep), decimals=2)
        self.median = np.round(np.median(self.dep), decimals=2)
        self.mean = np.round(np.mean(self.dep), decimals=2)
        self.t1 = self.axs[0].text(
            0.75,
            0.50,
            f"Dep. Max = {self.max} \nDep. Min = {self.min} \nDep. Median = {self.median}",
            transform=plt.gcf().transFigure,
        )

        self.sl_start = Slider(
            ax=self.ax_start,
            label="Dep. Ensemble",
            valmin=self.nmin,
            valmax=self.nmax,
            valinit=0,
            valfmt="%i",
            valstep=1,
        )

        self.sl_end = Slider(
            ax=self.ax_end,
            label="Rec. Ensemble",
            valmin=self.mmin,
            valmax=self.mmax,
            valinit=0,
            valfmt="%i",
            valstep=1,
        )

        self.sl_start.on_changed(self.update1)
        self.sl_end.on_changed(self.update2)
        self.button = Button(self.ax_button, "Save & Exit")
        # self.depminbutton = Button(self.ax_depminbutton, "<<")
        # self.depmaxbutton = Button(self.ax_depmaxbutton, ">>")
        # self.recminbutton = Button(self.ax_recminbutton, "<<")
        # self.recmaxbutton = Button(self.ax_recmaxbutton, ">>")

        self.button.on_clicked(self.exitwin)

    def update1(self, value):
        self.axs[0].scatter(self.x, self.dep, color="k")
        self.axs[0].scatter(self.x[0:value], self.dep[0:value], color="r")
        self.start_ens = value

    def update2(self, value):
        self.axs[1].scatter(self.x, self.dep, color="k")
        if value < 0:
            self.axs[1].scatter(
                self.x[self.n + value : self.n],
                self.dep[self.n + value : self.n],
                color="r",
            )
        self.end_ens = value

    def show(self):
        plt.show()

    def exitwin(self, event):
        plt.close()
        plt.close()


def trim_ends(vlobj, mask, method="Manual"):
    """
    Trims the start and end ensembles from ADCP data based on user selection.

    Args:
        vlobj: Object containing variable leader information (e.g., transducer depth).
        mask (numpy.ndarray): Mask array to update based on trimming.
        method (str, optional): Method for trimming, default is 'Manual'.

    Returns:
        numpy.ndarray: Updated mask array after trimming.

    Example:
        >>> from pyadps.utils import readrdi as rd
        >>> from profile_test import trim_ends
        >>> vlobj = rd.VariableLeader("data_file.000", run="fortran")
        >>> mask = np.zeros((100, 200))
        >>> mask = trim_ends(vlobj, mask)
    """
    transducer_depth = vlobj.vleader["Depth of Transducer"]
    # pressure = vlobj.vleader["Pressure"]
    if method == "Manual":
        out = PlotEnds(transducer_depth, delta=20)
        out.show()
        if out.start_ens > 0:
            mask[:, 0 : out.start_ens] = 1

        if out.end_ens < 0:
            mask[:, out.end_ens :] = 1

    return mask


def side_lobe_beam_angle(flobj, vlobj, mask, extra_cells=2):
    """
    Applies side-lobe interference correction based on beam angle and depth.

    Args:
        flobj: Object containing fixed leader information (e.g., beam angle, cell size).
        vlobj: Object containing variable leader information (e.g., transducer depth).
        mask (numpy.ndarray): Mask array to update based on side-lobe correction.
        extra_cells (int, optional): Additional cells to exclude. Default is 2.

    Returns:
        numpy.ndarray: Updated mask array after side-lobe correction.

    Example:
        >>> from pyadps.utils import readrdi as rd
        >>> from profile_test import side_lobe_beam_angle
        >>> flobj = rd.FixedLeader("data_file.000", run="fortran")
        >>> vlobj = rd.VariableLeader("data_file.000", run="fortran")
        >>> mask = np.zeros((100, 200))
        >>> mask = side_lobe_beam_angle(flobj, vlobj, mask)
    """
    beam_angle = int(flobj.system_configuration()["Beam Angle"])
    cell_size = flobj.field()["Depth Cell Len"]
    bin1dist = flobj.field()["Bin 1 Dist"]
    cells = flobj.field()["Cells"]
    ensembles = flobj.ensembles
    transducer_depth = vlobj.vleader["Depth of Transducer"]

    beam_angle = np.deg2rad(beam_angle)
    depth = transducer_depth / 10
    valid_depth = depth * np.cos(beam_angle) - bin1dist / 100
    valid_cells = np.trunc(valid_depth * 100 / cell_size) - extra_cells
    st.write(cells, valid_cells[100], extra_cells)

    for i in range(ensembles):
        c = int(valid_cells[i])
        if cells > c:
            mask[c:, i] = 1

    return mask


def side_lobe_rssi_bump(echo, mask):
    """
    Placeholder function for side-lobe correction based on RSSI bump.

    Args:
        echo (numpy.ndarray): Echo intensity data.
        mask (numpy.ndarray): Mask array to update.

    Returns:
        numpy.ndarray: Updated mask array.

    Note:
        This function is currently a placeholder and does not perform any operations.
    """
    pass


# read data
# filename = "BGS11000.000"
# fl = rd.FixedLeader(filename, run="fortran")
# vl = rd.VariableLeader(filename, run="fortran")
# echo = rd.echo(filename, run="fortran")
# vel = rd.velocity(filename, run="fortran")
#
# beam_angle = int(fl.system_configuration()["Beam Angle"])
# cell_size = fl.field()["Depth Cell Len"]
# bin1dist = fl.field()["Bin 1 Dist"]
# cells = fl.field()["Cells"]
#
# shape = np.shape(vel[0, :, :])
# mask = np.zeros(shape)
# orig_mask = np.copy(mask)
#
# mask = trim_ends(vl, mask)
# mask = side_lobe_beam_angle(fl, vl, mask)
# plotmask(orig_mask, mask)
# plotvar(echo, "Echo Intensity", mask)
