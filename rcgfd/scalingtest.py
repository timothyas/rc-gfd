import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

class ScalingTest():
    n_reservoir = 2000
    n_local     = 8
    n_workers   = 8
    batch_size  = 288 # 1 day
    n_nodes     = 1
    n_years     = 1
    sample      = 0

    @property
    def output_directory(self):
        odir = f"scaling-training/"
        odir += f"{self.n_reservoir:05d}res"
        odir += f"-{self.n_local:02d}loc"
        odir += f"-{self.n_workers:02d}work"
        odir += f"-{self.batch_size:05d}bs"
        odir += f"-{self.n_nodes:02d}node"
        odir += f"-{self.n_years:02d}year"
        odir += f"/sample-{self.sample:02d}"
        return odir

    @property
    def experiment(self):
        return self.output_directory.replace("scaling-","").replace("/","-").replace("-","_")

    @property
    def fname(self):
        return f"{self.output_directory}/memory_sampling.nc"

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            try:
                getattr(self, key)
            except KeyError:
                raise
            setattr(self, key, val)

    def convert2GiB(self, xda):
        xda *= 9.31e-10
        xda.attrs["units"] = "GiB"
        return xda

    def delta2minutes(self, xds):
        minutes = xds["delta_t"].values.astype("<m8[s]").astype(int)/60
        xds["minutes"] = xr.DataArray(minutes, xds["delta_t"].coords, xds["delta_t"].dims)
        xds = xds.swap_dims({"delta_t": "minutes"})
        return xds

    def store_memory_sampling(self, ms, name):

        # save a quick plot
        ms.plot(align=True)
        plt.savefig(f"{self.output_directory}/memory_sampling.pdf", bbox_inches="tight")

        # now store the data
        xds = self.ms2xda(ms, name)
        xds = xds.resample(delta_t="5s").mean()
        xds[name] = self.convert2GiB(xds[name])

        # expand dims
        xds = xds.expand_dims({
            "n_reservoir": [self.n_reservoir],
            "n_local": [self.n_local],
            "n_workers": [self.n_workers],
            "batch_size": [self.batch_size],
            "n_nodes": [self.n_nodes],
            "n_years": [self.n_years],
        })
        xds.to_netcdf(f"{self.output_directory}/memory_sampling.nc")


    @staticmethod
    def ms2xda(ms, name):

        table = ms.to_pandas()
        time = [np.datetime64(x) for x in table.index]
        delta_t = time - time[0]
        vals = [x for x in table[name].values]
        xda = xr.DataArray(vals, coords={'delta_t':delta_t}, dims=('delta_t',), name=name)
        return xda.to_dataset()

    def open_dataset(self, **kwargs):
        xds = xr.open_dataset(self.fname, **kwargs)

        # convert time to minutes
        xds = self.delta2minutes(xds)

        # get max memory and walltime
        with xr.set_options(keep_attrs=True):
            xds[f"max_mem"] = xds["training"].max("minutes")
        xds[f"walltime"] = xds["training"].minutes.where(~np.isnan(xds["training"])).max("minutes")
        xds[f"walltime"].attrs["units"] = "minutes"
        return xds
