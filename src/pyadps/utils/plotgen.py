import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox


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
