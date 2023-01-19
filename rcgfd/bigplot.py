"""Make a big plot grid..."""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

class BigPlot():

    z = 0
    vmax = 15
    vmin = -15
    time = 4800 * np.array([1, 3, 6])
    cmap = "cmo.balance"
    levels = 100
    n_cticks = 7

    prediction = "prediction"
    cbar_label=r"Potential Temperature Anomaly ($^\circ$C)"

    subplot_kw = {
        "figsize" :(14,18),
        "constrained_layout": True,
    }

    diff_t0 = True
    plot_truth = True

    @property
    def cticks(self):
        return np.linspace(self.vmin, self.vmax, self.n_cticks)


    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)



    def __call__(self, xds, row_dim="n_sub", **kwargs):

        nrows = len(xds[row_dim])
        nrows = nrows + 1 if self.plot_truth else nrows
        ncols = len(self.time)
        self.p_start = 1 if self.plot_truth else 0

        fig, axs = plt.subplots(nrows, ncols, **self.subplot_kw)

        mappables = []

        # truth
        if self.plot_truth:
            m = self._row_plot(xds["truth"], axr=axs[0,:])
            mappables.append(m)

        # now predictions, for each "row_dim" value
        for rd, axr in zip(xds[row_dim].values, axs[self.p_start:,:]):
            m = self._row_plot(
                    xds[self.prediction],
                    axr=axr,
                    dimsel={row_dim: rd})
            mappables.append(m)

        self._add_titles(axs)
        cbar = self._add_colorbar(fig, axs, mappables)

        return fig, axs

    def _row_plot(self, xda, axr, dimsel=None):

        if self.diff_t0:
            with xr.set_options(keep_attrs=True):
                diff = xda - xda.isel(time=0)
        else:
            diff = xda

        mappables = []
        for t, ax in zip(self.time, axr):

            plotme = diff.sel(z=self.z, time=t)
            plotme = plotme.sel(**dimsel) if dimsel is not None else plotme

            p = plotme.plot.contourf(
                ax=ax,
                add_colorbar=False,
                cmap=self.cmap,
                vmax=self.vmax,
                vmin=self.vmin,
                levels=self.levels)

            mappables.append(p)

            ax.set(xlabel="",ylabel="",title="",xticks=[],yticks=[])
            for key in ["bottom", "left"]:
                ax.spines[key].set_visible(False)

        label = xda.name.capitalize()
        if "truth" not in xda.name:
            label += "\n"
            dim = list(dimsel.keys())[0]
            val = list(dimsel.values())[0]
            key = xds[dim].attrs["label"] if "label" in xda[dim].attrs else dim
            label = label + key + f" = {val}"
            #label += r"$N_{sub}$ = %d" % int(n_sub)

        axr[0].set(ylabel=label)

        return mappables

    def _add_colorbar(self, fig, axs, mappables):

        cbar = fig.colorbar(
            mappables[self.p_start][-1],
            ax=axs,
            label=self.cbar_label,
            orientation="horizontal",
            pad=0.02,
            aspect=30,
            shrink=0.9,
            ticks=self.cticks,
        )
        cbar.ax.minorticks_off()
        return cbar

    def _add_titles(self, axs):
        for t, ax in zip(self.time, axs[0,:]):
            ax.set(title=r"$t = t_0 +$ %1.2f hours" % float(t/3600))
