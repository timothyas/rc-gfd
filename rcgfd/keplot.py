import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns
from typing import Callable


def plot_ke_relerr(relerr,
        hours=(1.33, 4, 8),
        cdim="n_sub",
        clabel=None,
        estimator="mean",
        errorbar=("ci", 99),
        show_persistence=False,
        persistence=None,
        fig=None,
        axs=None,
        **kwargs
        ):

    if clabel is None:
        if cdim == "n_sub":
            clabel = lambda n_sub : r"$N_{sub} = %d$" % n_sub
        else:
            clabel = ""

    if fig is None or axs is None:
        with plt.rc_context({"xtick.minor.size":4,"xtick.minor.width":1}):
            nrows = len(hours)
            width = nrows*4
            fig, axs = plt.subplots(1, nrows, figsize=(width,4), constrained_layout=True, sharex=True, sharey=True)

    color_start = 0 if cdim == "n_sub" else 3
    axs = [axs] if not isinstance(axs, (list, tuple, np.ndarray)) else axs
    n_lines = len(relerr[cdim])
    n_lines = n_lines+1 if show_persistence else n_lines
    for h, ax in zip(hours, axs):
        for i, d in enumerate(relerr[cdim].values):
            plotme = relerr.sel({cdim:d})
            plotme = plotme.sel(time=h*3600, method="nearest")
            _single_plot(plotme, ax=ax, estimator=estimator, errorbar=errorbar, label=clabel(d), color=f"C{color_start+i}", **kwargs)

        if show_persistence:
            _single_plot(persistence.sel(time=h*3600, method="nearest"), ax=ax, estimator=estimator, errorbar=errorbar, label="Persistence", color="gray", **kwargs)


        _cleanup_axis(fig, ax, hour=h, n_lines=n_lines)

    axs[0].set(ylabel="KE Density Relative Error")
    return fig, axs

def _single_plot(plotme, ax, **kwargs):
    plotme.name = "KE Density Relative Error"
    plotme=plotme.to_dataframe()
    sns.lineplot(
        data=plotme,
        x="k1d",
        y="KE Density Relative Error",
        ax=ax,
        **kwargs)
    return

def _cleanup_axis(fig, ax, hour, n_lines):
    # Label with time stamp
    ax.text(
        1e-2, 1.,"$t = t_0 + %1.2f$ hrs" % float(hour),
        ha="right",
        va="top",
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
            ncol=n_lines,
            frameon=True,
        )
        # Make legend handle linewidth bigger
        [l.set_linewidth(3) for l in leg.legendHandles]
