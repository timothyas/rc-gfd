import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns

def plot_ke_relerr(relerr, errorbar="ci", fig=None, axs=None):

    if fig is None or axs is None:
        with plt.rc_context({"xtick.minor.size":4,"xtick.minor.width":1}):
            fig, axs = plt.subplots(1, 3, figsize=(12,4), constrained_layout=True, sharex=True, sharey=True)

    for t, ax in zip(relerr["time"], axs):
        for n_sub in relerr["n_sub"].values:
            plotme = relerr.sel(n_sub=n_sub,time=t)
            plotme=plotme.to_dataframe()
            sns.lineplot(
                data=plotme,
                x="k1d",
                y="KE Density",
                ax=ax,
                label=r"$N_{sub} = %d$" % n_sub,
                errorbar=errorbar,
            )

        # Label with time stamp
        ax.text(
            8e-4, 0.9,"$t = t_0 + %1.2f$ hrs" % float(t/3600),
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
            ylim=[0,1.1],
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
