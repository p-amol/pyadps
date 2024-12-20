import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox

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


class PlotEnds:
    def __init__(self, pressure, delta=10):
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



class PlotNoise:
    def __init__(self, echo):
        self.cutoff = 0
        self.echo = echo

        # Assign axes for plots and widgets
        self.fig, self.axs = plt.subplots(1, 2)
        self.fig.set_facecolor("darkgrey")

        plt.subplots_adjust(bottom=0.28, right=0.72)
        self.ax_tbox = self.fig.add_axes(rect=(0.85, 0.8, 0.1, 0.05))
        self.ax_end = self.fig.add_axes(rect=(0.25, 0.1, 0.47, 0.03))
        self.ax_start = self.fig.add_axes(rect=(0.25, 0.15, 0.47, 0.03))
        self.ax_button = self.fig.add_axes(rect=(0.81, 0.65, 0.15, 0.075))

        # Displays cutoff value
        self.textvar = self.axs[0].text(
            0.78,
            0.75,
            f"Default Cutoff: {self.cutoff}",
            color="blue",
            transform=plt.gcf().transFigure,
        )

        # Plot echo for first and last ensemble
        shape = np.shape(echo)
        self.x = np.arange(0, shape[1], 1)

        self.l1 = [
            self.axs[0].plot(echo[i, :, 0], self.x, label=f"Beam {i+1}")
            for i in range(4)
        ]

        self.l2 = [
            self.axs[1].plot(echo[i, :, -1], self.x, label=f"Beam {i+1}")
            for i in range(4)
        ]

        # Figure Labels
        for i in range(2):
            self.axs[i].legend()
            self.axs[i].set_xlabel("Echo")
            self.axs[i].set_xlim([20, 200])
        self.axs[0].set_ylabel("Cell")
        self.fig.suptitle("Noise Floor Identification")

        # Display statistics
        self.axs[0].text(0.82, 0.60, "Statistics", transform=plt.gcf().transFigure)
        l1max = np.max(echo[0, :, :])
        l1min = np.min(echo[0, :, :])
        l1med = np.median(echo[0, :, :])
        self.t1 = self.axs[0].text(
            0.75,
            0.50,
            f"Dep. Max = {l1max} \nDep. Min = {l1min} \nDep. Median = {l1med}",
            transform=plt.gcf().transFigure,
        )

        l2max = np.max(echo[:, :, -1])
        l2min = np.min(echo[:, :, -1])
        l2med = np.median(echo[:, :, -1])
        self.t2 = self.axs[0].text(
            0.75,
            0.35,
            f"Rec. Max = {l2max} \nRec. Min = {l2min}\nRec. Median = {l2med}",
            transform=plt.gcf().transFigure,
        )

        # Define Widgets
        self.tbox = TextBox(
            ax=self.ax_tbox,
            label="Enter Cutoff",
            color="lightgrey",
            hovercolor="yellow",
            initial="0",
        )

        self.sl_start = Slider(
            ax=self.ax_start,
            label="Deployment Ensemble",
            valmin=0,
            valmax=10,
            valinit=0,
            valfmt="%i",
            valstep=1,
        )
        self.sl_end = Slider(
            ax=self.ax_end,
            label="Recovery Ensemble",
            valmin=-11,
            valmax=-1,
            valinit=0,
            valfmt="%i",
            valstep=1,
        )

        self.button = Button(self.ax_button, "Save & Exit")

        # Activate widgets
        self.sl_start.on_changed(self.update1)
        self.sl_end.on_changed(self.update2)
        self.tbox.on_submit(self.submit)
        self.button.on_clicked(self.exitwin)

    def update1(self, value):
        for i, line in enumerate(self.l1):
            line[0].set_xdata(self.echo[i, :, value])

        self.t1.remove()
        l1max = np.max(self.echo[:, :, value])
        l1min = np.min(self.echo[:, :, value])
        l1med = np.median(self.echo[:, :, value])
        self.t1 = self.axs[0].text(
            0.75,
            0.50,
            f"Dep. Max = {l1max} \nRec. Min = {l1min}\nRec. Median = {l1med}",
            transform=plt.gcf().transFigure,
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update2(self, value):
        for i, line in enumerate(self.l2):
            line[0].set_xdata(self.echo[i, :, value])
        self.t2.remove()
        l2max = np.max(self.echo[:, :, value])
        l2min = np.min(self.echo[:, :, value])
        l2med = np.median(self.echo[:, :, value])
        self.t2 = self.axs[0].text(
            0.75,
            0.35,
            f"Rec. Max = {l2max} \nRec. Min = {l2min}\nRec. Median = {l2med}",
            transform=plt.gcf().transFigure,
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def submit(self, exp):
        try:
            self.cutoff = int(exp)
            self.textvar.remove()
            self.textvar = self.axs[0].text(
                0.78,
                0.75,
                f"Cutoff:{self.cutoff}",
                color="black",
                transform=plt.gcf().transFigure,
            )
        except ValueError:
            self.cutoff = 0
            self.textvar.remove()
            self.textvar = self.axs[0].text(
                0.78,
                0.75,
                "Error: Enter an integer",
                color="red",
                transform=plt.gcf().transFigure,
            )

    def show(self):
        plt.show()

    def exitwin(self, event):
        plt.close()


def plotmask(mask1, mask2):
    cpal = "binary"
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    shape = np.shape(mask1)
    x = np.arange(0, shape[1])
    y = np.arange(0, shape[0])
    X, Y = np.meshgrid(x, y)
    axs[0].pcolor(X, Y, mask1, cmap=cpal, label="Original Mask")
    axs[1].pcolor(X, Y, mask2, cmap=cpal, label="New Mask")
    axs[0].set_title("Original Mask")
    axs[1].set_title("New Mask")
    plt.xlabel("Ensembles")
    plt.ylabel("Cells")
    fig.tight_layout()
    plt.show()


def plotvar(var, name, mask=None, alpha=True):

    shape = np.shape(var)
    cpal = "turbo"
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    x = np.arange(0, shape[-1])
    y = np.arange(0, shape[1])
    X, Y = np.meshgrid(x, y)
    i = 0
    for j in range(2):
        for k in range(2):
            if mask is not None:
                if alpha:
                    nanmask = np.copy(mask)
                    nanmask[mask == 0] = np.nan
                    axs[j, k].pcolor(X, Y, var[i, :, :], cmap=cpal)
                    axs[j, k].pcolor(X, Y, nanmask, cmap="binary", alpha=0.05)
                else:
                    maskdata = np.ma.masked_array(var[i, :, :], mask)
                    axs[j, k].pcolor(X, Y, maskdata, cmap=cpal)
            else:
                axs[j, k].pcolor(X, Y, var[i, :, :], cmap=cpal)

            axs[j, k].set_title(f"Beam {i+1}")
            axs[j, k].set_xlabel("Ensembles")
            axs[j, k].set_ylabel("Cells")
            i = i + 1
    fig.suptitle(name)
    fig.tight_layout()
    plt.show()


def plot1d(data):
    fig = plt.figure(figsize=(2, 2), facecolor="lightskyblue", layout="constrained")
    fig.suptitle("Fixed Leader Data")
    ax = fig.add_subplot()
    ax.set_title("Fleader", loc="center", fontstyle="oblique", fontsize="medium")
    ax.set_xlabel("Ensemble")
    ax.set_ylabel("Fleader")
    (l,) = ax.plot(data)
    l.set_color("C0")
