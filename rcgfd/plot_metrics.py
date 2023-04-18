import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsPlot():

    metrics     = ("nrmse", "ke_rmse")
    errorbar    = ("ci", 99)
    cdim        = "n_sub"
    cdim_label  = None
    time        = np.arange(0, 12*3600+1, 4800)
    show_persistence = True
    ax_size     = (4.5, 4)

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            try:
                getattr(self, key)
            except:
                raise

            setattr(self, key, val)

        if self.cdim_label is None:
            if self.cdim == "n_sub":
                self.cdim_label = self.n_sub_label
            elif self.cdim == "gamma":
                self.cdim_label = self.gamma_label
            else:
                self.cdim_label = lambda x : ""


    @staticmethod
    def n_sub_label(n_sub):
        return r"$N_{sub} = %d$" % n_sub


    @staticmethod
    def gamma_label(gamma):
        if gamma < 1e-16:
            return r"$\gamma = 0$"
        else:
            return r"$\gamma = 10^{%d}$" % np.log10(gamma)


    @staticmethod
    def ylabel(metric):

        if metric == "nrmse":
            return "NRMSE"

        elif metric =="acc":
            return "ACC"

        elif metric == "ke_rmse":
            return "KE Density RMSE"

        elif metric == "ke_nrmse":
            return "KE Density NRMSE"

        else:
            return metric


    @property
    def color_start(self):
        return 0 if self.cdim == "n_sub" else 3


    def __call__(self, xds, show_time, **kwargs):

        ncols = len(self.metrics)
        width = self.ax_size[0]*ncols
        fig, axs = plt.subplots(1, ncols, figsize=(width, self.ax_size[1]), constrained_layout=True)

        plot = self.plot_vs_time if show_time else self.plot

        for metric, ax in zip(self.metrics, axs):
            plot(xds[metric], xds[f"p_{metric}"], ax, **kwargs)
            ax.set(ylabel=self.ylabel(metric))

        return fig, axs


    def plot_vs_time(self, xda, pda, ax, **kwargs):

        errorbar = kwargs.pop("errorbar", self.errorbar)

        for i, dim in enumerate(xda[self.cdim].values):
            plotme = xda.sel({self.cdim:dim})

            # If any sample hits inf, remove it...
            # it's not clear what the statistics mean at this point
            # all that matters is that this setting is unreliable.
            tinf = plotme.time.where(np.isinf(plotme).any("sample")).min("time")
            plotme = plotme.sel(time=slice(tinf.values))

            df = plotme.to_dataframe().reset_index()
            sns.lineplot(
                data=df,
                x="time",
                y=xda.name,
                ax=ax,
                label=self.cdim_label(dim),
                errorbar=errorbar,
                color=f"C{i+self.color_start}" if "palette" not in kwargs else None,
                **kwargs
            )
        if self.show_persistence:
            sns.lineplot(
                    data=pda.to_dataframe().reset_index(),
                    x="time",
                    y=pda.name,
                    ax=ax,
                    label="Persistence",
                    errorbar=errorbar,
                    color="gray",
                    **kwargs)

        ax.set(
            xticks=3600*np.array([0,4,8,12]),
            xticklabels=[0,4,8,12])

        return


    def plot(self, xda, pda, ax, **kwargs):

        showfliers = kwargs.pop("showfliers", False)
        palette = kwargs.pop("palette",
                [f"C{i+self.color_start}" for i in range(len(xda[self.cdim]))])

        plotme = (xda.sel(time=self.time)**2).mean('time')
        plotme = np.sqrt(plotme)
        df = plotme.to_dataframe().reset_index()
        sns.violinplot(
            data=df,
            y=xda.name,
            x=self.cdim,
            ax=ax,
            showfliers=showfliers,
            palette=palette,
            **kwargs
        )

        if self.show_persistence:
            plotme = np.sqrt( (pda**2).mean("time") )
            ax.axhline( plotme.median("sample"), color="gray")
        return
