import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsPlot():

    metrics     = ("nrmse", "ke_rmse")
    errorbar    = ("ci", 99)
    cdim        = "n_sub"
    cdim_label  = None
    time        = np.arange(0, 12*3600+1, 4800)

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
        width = 4.5*ncols
        fig, axs = plt.subplots(1, ncols, figsize=(width,4), constrained_layout=True)

        plot = self.plot_vs_time if show_time else self.plot

        for metric, ax in zip(self.metrics, axs):
            plot(xds[metric], ax, **kwargs)
            ax.set(ylabel=self.ylabel(metric))

        return fig, axs


    def plot_vs_time(self, xda, ax, **kwargs):

        errorbar = kwargs.pop("errorbar", self.errorbar)

        for i, dim in enumerate(xda[self.cdim].values):
            plotme = xda.sel({self.cdim:dim})
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

        ax.set(
            xticks=3600*np.array([0,4,8,12]),
            xticklabels=[0,4,8,12])

        return


    def plot(self, xda, ax, **kwargs):

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
        return
