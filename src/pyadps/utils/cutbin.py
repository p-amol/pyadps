import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, RectangleSelector, Slider

from pyadps.utils import readrdi as rd


class CutBins:
    def __init__(
        self, data, mask, newmask=False, t1=0, t2=200, tinc=500, z1=0, z2=-1, zinc=0
    ):
        # DATA SETUP
        self.orig_data = data
        self.orig_shape = np.shape(self.orig_data)
        self.fill = 999
        self.maskarray = mask
        if not newmask:
            self.orig_data[self.maskarray == 1] = self.fill

        self.t1, self.t2, self.tinc = t1, t2, tinc
        self.z1, self.z2, self.zinc = z1, z2, zinc
        if z2 == -1:
            self.z2 = self.orig_shape[0]

        self.data = self.orig_data[self.z1 : self.z2, self.t1 : self.t2]
        self.orig_subset = self.orig_data[self.z1 : self.z2, self.t1 : self.t2]
        self.datacopy = np.copy(self.orig_data)
        self.datamin = np.min(self.orig_data)
        self.datamax = np.max(self.orig_data)
        self.shape = np.shape(self.data)

        # PLOT SETUP
        self.t = np.arange(self.t1, self.t2)
        self.z = np.arange(self.z1, self.z2)
        self.tickinterval = int((self.t2 - self.t1) / 5)
        self.xticks = np.arange(self.t1, self.t2, self.tickinterval)
        self.X, self.Y = np.meshgrid(self.t, self.z)
        self.fig, self.axs = plt.subplot_mosaic(
            [["a", "b"], ["c", "b"]],
            figsize=(12, 10),
            width_ratios=[2, 1],
            height_ratios=[1.75, 1],
        )
        self.fig.set_facecolor("darkgrey")
        plt.subplots_adjust(top=0.82, right=0.95)

        # ADDING WIDGET AXES
        self.ax_clear_button = self.fig.add_axes(rect=(0.125, 0.90, 0.08, 0.025))
        self.ax_delete_button = self.fig.add_axes(rect=(0.225, 0.90, 0.08, 0.025))
        self.ax_refill_button = self.fig.add_axes(rect=(0.325, 0.90, 0.08, 0.025))
        self.ax_next_button = self.fig.add_axes(rect=(0.630, 0.65, 0.02, 0.050))
        self.ax_previous_button = self.fig.add_axes(rect=(0.075, 0.65, 0.02, 0.050))
        self.ax_radio_button = self.fig.add_axes(rect=(0.725, 0.87, 0.10, 0.10))
        self.ax_exit_button = self.fig.add_axes(rect=(0.825, 0.025, 0.08, 0.035))
        self.ax_hslider = self.fig.add_axes(rect=(0.125, 0.85, 0.50, 0.03))
        self.ax_vslider = self.fig.add_axes(rect=(0.04, 0.25, 0.03, 0.50))

        self.ax_delete_button.set_visible(False)
        self.ax_refill_button.set_visible(False)

        # --- Slider settings ---
        # Initial slider settings
        self.hevent = 0
        self.vevent = 0

        # Slider options
        self.hslider = Slider(
            ax=self.ax_hslider,
            label="Ensemble",
            valmin=self.t1,
            valmax=self.t2,
            valinit=self.hevent,
            valfmt="%i",
            valstep=1,
        )

        self.vslider = Slider(
            ax=self.ax_vslider,
            label="Bins",
            valmin=self.z1,
            valmax=self.z2,
            valinit=self.vevent,
            valfmt="%i",
            valstep=1,
            orientation="vertical",
        )

        # Button Labels
        self.clear_button = Button(self.ax_clear_button, "Clear")
        self.delete_button = Button(self.ax_delete_button, "Delete")
        self.refill_button = Button(self.ax_refill_button, "Refill")
        self.previous_button = Button(self.ax_previous_button, "<")
        self.next_button = Button(self.ax_next_button, ">")
        self.exit_button = Button(self.ax_exit_button, "Save & Exit")
        # self.cell_button = Button(self.ax_cell_button, "Cell")
        # self.ensemble_button = Button(self.ax_ensemble_button, "Ensemble")
        self.radio_button = RadioButtons(
            self.ax_radio_button, ("Bin", "Ensemble", "Cell", "Region")
        )

        # --------------PLOTS---------------------

        # Settings colorbar extreme to black
        cmap = mpl.cm.turbo.with_extremes(over="k")
        # FILL PLOT
        self.mesh = self.axs["a"].pcolormesh(
            self.X, self.Y, self.data, cmap=cmap, picker=True, vmin=0, vmax=255
        )
        plt.colorbar(self.mesh, orientation="horizontal")
        self.axs["a"].set_xlim([self.t1, self.t2])
        self.axs["a"].set_ylim([self.z1, self.z2])
        # Draw vertical and horizontal lines
        (self.vline,) = self.axs["a"].plot(
            [self.t1, self.t1], [self.z1, self.z2], color="r", linewidth=2.5
        )
        (self.hline,) = self.axs["a"].plot(
            [self.t1, self.t2], [self.z1, self.z1], color="r", linewidth=2.5
        )

        # PROFILE
        (self.profile,) = self.axs["b"].plot(
            self.data[self.z1 : self.z2, self.t1 + self.hevent], range(self.z1, self.z2)
        )

        self.axs["b"].set_xlim([self.datamin, self.datamax])
        self.profile_text = self.axs["b"].text(
            0.95,
            0.95,
            f"Ensemble No.: {self.t1 + self.hevent}",
            verticalalignment="bottom",
            horizontalalignment="right",
            transform=self.axs["b"].transAxes,
            color="k",
            fontsize=12,
        )

        # TIME SERIES
        (self.tseries,) = self.axs["c"].plot(
            range(self.t1, self.t2), self.data[self.z1 + self.vevent, self.t1 : self.t2]
        )
        self.axs["c"].set_ylim([self.datamin, self.datamax])
        self.tseries_text = self.axs["c"].text(
            0.90,
            0.90,
            f"Bin No.: {self.z1 + self.vevent}",
            verticalalignment="bottom",
            horizontalalignment="right",
            transform=self.axs["c"].transAxes,
            color="k",
            fontsize=12,
        )
        # --------------END PLOTS---------------------

        # EVENTS
        self.onclick = self.onclick_bin
        self.hslider.on_changed(self.hupdate)
        self.vslider.on_changed(self.vupdate)
        self.clear_button.on_clicked(self.clear)
        self.radio_button.on_clicked(self.radio)
        self.cid = self.fig.canvas.mpl_connect("pick_event", self.onclick)

        self.delete_button.on_clicked(self.boxdelete)
        self.refill_button.on_clicked(self.boxrefill)
        self.next_button.on_clicked(self.next)
        self.previous_button.on_clicked(self.previous)
        self.exit_button.on_clicked(self.exit)

    def next(self, event):
        if self.t2 <= self.orig_shape[1]:
            # Next works till the last subset. The if statement checks for last subset.
            self.t1 = self.t1 + self.tinc
            self.t2 = self.t2 + self.tinc
            if self.t2 > (self.orig_shape[1]):
                # If in last subset create a dummy data set with missing value.
                self.data = self.datacopy[
                    self.z1 : self.z2, self.t1 : self.orig_shape[1]
                ]
                self.orig_subset = self.orig_data[
                    self.z1 : self.z2, self.t1 : self.orig_shape[1]
                ]
                self.missing = (
                    np.ones((self.z2 - self.z1, self.t2 - self.orig_shape[1]))
                    * self.fill
                )
                # self.data consist of data along with flagged value
                self.data = np.append(self.data, self.missing, axis=1)
                # self.orig_subset contains only the subset of the original data
                # Useful for plotting time series and profiles
                self.orig_subset = np.append(self.orig_subset, self.missing, axis=1)
            else:
                self.data = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
                self.orig_subset = self.orig_data[self.z1 : self.z2, self.t1 : self.t2]

            self.mesh.set_array(self.data)
            self.tick = np.arange(self.t1, self.t2, self.tickinterval)
            self.axs["a"].set_xticks(self.xticks, self.tick)

            self.profile.set_xdata(self.orig_subset[:, self.hevent])
            self.profile_text.set_text(f"Ensemble No.: {self.t1 + self.hevent}")
            self.vline.set_xdata([self.hevent, self.hevent])

            self.tseries.set_ydata(self.orig_subset[self.vevent, :])
            self.tseries_text.set_text(f"Bin No.: {self.z1 + self.vevent}")
            self.hline.set_ydata([self.vevent, self.vevent])

            self.fig.canvas.draw()

    def previous(self, event):
        if self.t1 >= self.tinc:
            self.t1 = self.t1 - self.tinc
            self.t2 = self.t2 - self.tinc
            self.tick = np.arange(self.t1, self.t2, self.tickinterval)
            self.data = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
            self.axs["a"].set_xticks(self.xticks, self.tick)
            self.mesh.set_array(self.data)

            # Reset sliders
            self.profile.set_xdata(self.orig_data[self.z1 : self.z2, self.hevent])
            self.profile_text.set_text(f"Ensemble No.: {self.hevent}")
            self.vline.set_xdata([self.hevent, self.hevent])

            self.tseries.set_ydata(self.orig_data[self.vevent, self.t1 : self.t2])
            self.tseries_text.set_text(f"Bin No.: {self.z1 + self.vevent}")
            self.hline.set_ydata([self.vevent, self.vevent])

            self.fig.canvas.draw()

    def radio(self, event):
        self.fig.canvas.mpl_disconnect(self.cid)
        if event == "Bin":
            self.cid = self.fig.canvas.mpl_connect("pick_event", self.onclick_bin)
        elif event == "Ensemble":
            self.cid = self.fig.canvas.mpl_connect("pick_event", self.onclick_ens)
        elif event == "Cell":
            self.cid = self.fig.canvas.mpl_connect("pick_event", self.onclick_cell)
        else:
            self.rid = RectangleSelector(
                self.axs["a"],
                self.onclick_box,
                useblit=True,
                minspanx=2,
                minspany=2,
                interactive=True,
            )

    def clear(self, event):
        if event.button == 1:
            self.datacopy = np.copy(self.orig_data)
            if self.t2 >= (self.orig_shape[1]):
                test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
                test = np.append(test, self.missing, axis=1)
            else:
                test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]

            # self.mesh.set_array(self.datacopy[self.z1 : self.z2, self.t1 : self.t2])
            self.mesh.set_array(test)
            self.fig.canvas.draw()

    def hupdate(self, event):
        self.hevent = event
        self.profile.set_xdata(self.orig_subset[:, self.hevent])
        self.profile_text.set_text(f"Ensemble No.: {self.t1 + self.hevent}")
        self.vline.set_xdata([self.hevent, self.hevent])

    def vupdate(self, event):
        self.vevent = event
        self.tseries.set_ydata(self.orig_subset[self.vevent, :])
        self.tseries_text.set_text(f"Bin No.: {self.z1 + self.vevent}")
        self.hline.set_ydata([self.vevent, self.vevent])

    def onclick_bin(self, event):
        ind = event.ind
        x = ind // (self.t[-1] + 1)
        # y = ind % (self.t[-1] + 1)
        xx = self.z1 + x
        # yy = self.t1 + y
        if np.all(self.datacopy[xx, :] == self.fill):
            self.datacopy[xx, :] = np.copy(self.orig_data[xx, :])

        else:
            self.datacopy[xx, :] = self.fill

        if self.t2 >= (self.orig_shape[1]):
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
            test = np.append(test, self.missing, axis=1)
        else:
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]

        # self.mesh.set_array(self.datacopy[self.z1 : self.z2, self.t1 : self.t2])
        self.mesh.set_array(test)
        self.hline.set_ydata([x, x])
        self.vslider.set_val(x[0])
        self.fig.canvas.draw()

    def onclick_ens(self, event):
        ind = event.ind
        if np.size(ind) != 1:
            return
        # x = ind // (self.t[-1] + 1)
        y = ind % (self.t[-1] + 1)
        yy = self.t1 + y

        if yy < self.orig_shape[1]:
            if np.all(self.datacopy[:, yy] == self.fill):
                self.datacopy[:, yy] = np.copy(self.orig_data[:, yy])
            else:
                self.datacopy[:, yy] = self.fill

        if self.t2 >= (self.orig_shape[1]):
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
            test = np.append(test, self.missing, axis=1)
        else:
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
        # self.mesh.set_array(self.datacopy[self.z1 : self.z2, self.t1 : self.t2])
        self.mesh.set_array(test)
        self.hline.set_xdata([y, y])
        self.hslider.set_val(y[0])
        self.fig.canvas.draw()

    def onclick_cell(self, event):
        ind = event.ind
        if np.size(ind) != 1:
            return
        x = ind // (self.t[-1] + 1)
        y = ind % (self.t[-1] + 1)
        xx = self.z1 + x
        yy = self.t1 + y

        if yy < self.orig_shape[1]:
            if self.datacopy[xx, yy] == self.fill:
                self.datacopy[xx, yy] = np.copy(self.orig_data[x, y])
            else:
                self.datacopy[xx, yy] = self.fill

        if self.t2 > (self.orig_shape[1]):
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
            test = np.append(test, self.missing, axis=1)
        else:
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]

        # self.mesh.set_array(self.datacopy[self.z1 : self.z2, self.t1 : self.t2])
        self.mesh.set_array(test)
        self.vline.set_xdata([y, y])
        self.hline.set_ydata([x, x])
        self.hslider.set_val(y[0])
        self.vslider.set_val(x[0])
        self.fig.canvas.draw()

    def onclick_box(self, eclick, erelease):
        self.ax_delete_button.set_visible(True)
        self.ax_refill_button.set_visible(True)
        plt.gcf().canvas.draw()
        self.x11, self.y11 = int(eclick.xdata), int(eclick.ydata)
        self.x22, self.y22 = int(erelease.xdata) + 1, int(erelease.ydata) + 1

        print(
            f"({self.x11:3.2f}, {self.y11:3.2f}) --> ({self.x22:3.2f}, {self.y22:3.2f})"
        )
        print(f" The buttons you used were: {eclick.button} {erelease.button}")

    def boxdelete(self, event):
        z1 = self.z1 + self.y11 + 1
        z2 = self.z1 + self.y22
        t1 = self.t1 + self.x11 + 1
        t2 = self.t1 + self.x22
        self.datacopy[z1:z2, t1:t2] = self.fill

        if self.t2 > (self.orig_shape[1]):
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
            test = np.append(test, self.missing, axis=1)
        else:
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]

        # self.mesh.set_array(self.datacopy[self.z1 : self.z2, self.t1 : self.t2])
        self.mesh.set_array(test)
        self.fig.canvas.draw()

    def boxrefill(self, event):
        z1 = self.z1 + self.y11 + 1
        z2 = self.z1 + self.y22
        t1 = self.t1 + self.x11 + 1
        t2 = self.t1 + self.x22
        self.datacopy[z1:z2, t1:t2] = self.orig_data[z1:z2, t1:t2]

        if self.t2 > (self.orig_shape[1]):
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
            test = np.append(test, self.missing, axis=1)
        else:
            test = self.datacopy[self.z1 : self.z2, self.t1 : self.t2]
        # self.mesh.set_array(self.datacopy[self.z1 : self.z2, self.t1 : self.t2])
        self.mesh.set_array(test)
        self.fig.canvas.draw()

    def exit(self, event):
        plt.close()

    def mask(self):
        self.maskarray[self.datacopy == self.fill] = 1
        return self.maskarray


# filename = "BGS11000.000"
# ds = rd.echo(filename, run="fortran")
# echo = ds[0, :, :]
# shape = echo.shape
# mask = np.zeros(shape)

# manual = CutBins(echo, mask)
# plt.show()
# mask = manual.mask()
# plt.pcolormesh(mask)
# plt.show()
