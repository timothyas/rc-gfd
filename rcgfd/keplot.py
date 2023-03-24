import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns


def plot_ke_relerr(relerr,
        hours=(1.33, 4, 8),
        cdim="n_sub",
        clabel=None,
        errorbar=("ci", 99),
        fig=None,
        axs=None):

    if clabel is None:
        if cdim == "n_sub":
            clabel = lambda n_sub : r"$N_{sub} = %d$" % n_sub
        else:
            clabel = lambda x : ""

    if fig is None or axs is None:
        with plt.rc_context({"xtick.minor.size":4,"xtick.minor.width":1}):
            nrows = len(hours)
            width = nrows*4
            fig, axs = plt.subplots(1, nrows, figsize=(width,4), constrained_layout=True, sharex=True, sharey=True)

    color_start = 0 if cdim == "n_sub" else 3
    axs = [axs] if not isinstance(axs, (list, tuple, np.ndarray)) else axs
    for t, ax in zip(hours, axs):
        for i, d in enumerate(relerr[cdim].values):
            plotme = relerr.sel({cdim:d})
            plotme = plotme.sel(time=t*3600, method="nearest")
            plotme.name = "KE Density Relative Error"
            plotme=plotme.to_dataframe()
            sns.lineplot(
                data=plotme,
                x="k1d",
                y="KE Density Relative Error",
                ax=ax,
                label=clabel(d),
                errorbar=errorbar,
                color=f"C{color_start+i}",
            )

        # Label with time stamp
        ax.text(
            8e-4, 0.9,"$t = t_0 + %1.2f$ hrs" % float(t),
            ha="center",
            va="center",
            transform=ax.transData,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round,pad=.5",
            },
        )
        leg = ax.legend()
        leg.remove()

        # Log with minor axes
        ax.set(
            xscale="log",
            ylim=[-1,1.1],
            ylabel="",
            xlabel="",
        )
        ax.xaxis.set_minor_locator(LogLocator(numticks=999, subs=(.2, .4, .6, .8)))
        ax.grid(True, which="both")

        ax.set(
            xlabel=r"Wavenumber, $|\mathbf{K}|$ (rad km$^{-1}$)",
        )

        # Add legend
        if ax.get_subplotspec().is_first_col():
            leg = fig.legend(
                loc="center",
                bbox_to_anchor=(0.5,-0.1),
                ncol=4,
                frameon=True,
            )
            # Make legend handle linewidth bigger
            [l.set_linewidth(3) for l in leg.legendHandles]
    axs[0].set(ylabel="KE Density Relative Error")
    return fig, axs
