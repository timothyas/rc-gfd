import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

from .bigplot import BigPlot

class SingleTimePlot(BigPlot):
    time = 12*3600
    subplot_kw = {
        "figsize": (18,5.5),
        "constrained_layout": True,
    }

    def __call__(self, xds, col_dim="n_sub", **kwargs):

        nrows = 1
        ncols = len(xds[col_dim])+1 if self.plot_truth else len(xds[col_dim])
        self.p_start = 1 if self.plot_truth else 0

        fig, axs = plt.subplots(nrows, ncols, **self.subplot_kw)
        axs = [axs] if not isinstance(axs, (list, np.ndarray)) else axs

        mappables = []

        # truth
        if self.plot_truth:
            m = self._plot(xds["truth"], ax=axs[0])
            mappables.append(m)

        for cd, ax in zip(xds[col_dim].values, axs[self.p_start:]):
            m = self._plot(xds[self.prediction], ax=ax, dimsel={col_dim: cd})
            mappables.append(m)

        self._add_titles(axs, xds[col_dim])
        cbar = self._add_colorbar(fig, axs, mappables)

        return fig, axs

    def _plot(self, xda, ax, dimsel=None):

        if self.diff_t0:
            with xr.set_options(keep_attrs=True):
                plotme = xda - xda.isel(time=0)

        else:
            plotme = xda

        plotme = plotme.sel(z=self.z, time=self.time)
        plotme = plotme.sel(**dimsel) if dimsel is not None else plotme
        p = plotme.plot.contourf(
            x="x",
            ax=ax,
            add_colorbar=False,
            cmap=self.cmap,
            vmax=self.vmax,
            vmin=self.vmin,
            levels=self.levels,
        )

        ax.set(xlabel="", title="", ylabel="", xticks=[], yticks=[])
        for key in ["bottom", "left"]:
            ax.spines[key].set_visible(False)

        return p

    def _add_titles(self, axs, col_dim):

        labels = ["Truth"] if self.plot_truth else []

        for v in col_dim.values:
            label = f"{col_dim.name} = {v}"
            labels.append(label)

        for label, ax in zip(labels,axs):
            ax.set(title=label)
        return


    def _add_colorbar(self, fig, axs, mappables):
        cbar = fig.colorbar(
            mappables[-1],
            ax=axs,
            label=self.cbar_label,
            orientation="horizontal",
            pad=0.05,
            aspect=40,
            shrink=0.8,
            ticks=self.cticks,
        )
        cbar.ax.minorticks_off()
        return cbar
