import numpy as np
import xarray as xr

from ddc import YAMLParser
from sqgtools import XSQGTurb

class Dataset():

    n_sub       = None
    n_samples   = 50

    chunks      = None
    dims        = ("n_sub",)
    time        = np.arange(0, 12*3600+1, 4800)
    include_persist = True

    squeeze     = True

    def __init__(self, **kwargs):
        for key,val in kwargs.items():
            try:
                getattr(self, key)
            except:
                raise
            setattr(self, key, val)

        self.chunks = {"x":16, "y":16} if self.chunks is None else self.chunks
        self.n_sub = (self.n_sub,) if not isinstance(self.n_sub, (list, tuple)) else self.n_sub


    def postprocess_dataset(self, xds):

        if self.squeeze:
            xds = xds.squeeze()
        xds.attrs['keep_spinup'] = 'False'
        xds.attrs['spinup_steps_predict'] = 0

        for key in self.dims:
            if key in xds["truth"].dims:
                xds["truth"] = xds["truth"].isel({key: 0})


        xds = self.calc_metrics(xds)
        xds = self.calc_spectral_metrics(xds)

        # Persistence error
        xds["persistence"] = xds["truth"].isel(time=0)
        pds = self.calc_metrics(xds[["truth","persistence"]], pkey="persistence")
        pds = self.calc_spectral_metrics(pds, pkey="persistence")
        for key in ["rmse", "nrmse", "acc", "ke_rel_err", "ke_rmse", "ke_nrmse"]:
            xds[f"p_{key}"] = pds[key]
        return xds


    def renormalize_dataset(self, xds, fname):

        yp = YAMLParser()
        c = yp.read(fname.replace("results.zarr", "config-lazy.yaml"))
        if "preprocessing" in c:
            sd = c["preprocessing"].pop("norm_factor", None)

        if sd is not None:
            with xr.set_options(keep_attrs=True):
                for key in ["truth", "prediction"]:
                    xds[key] = xds[key]*sd
        return xds


    def calc_metrics(self, xds, pkey="prediction"):

        xds["error"] = xds[pkey] - xds["truth"]
        xds["absolute_error"] = np.abs(xds["error"])

        dims = ["x","y","z"]
        xds["rmse"] = np.sqrt( (xds["error"]**2).mean(dims) )
        xds["rmse"].attrs = {
            "label": "RMSE",
        }
        xds["nrmse"] = xds["rmse"] / xds["truth"].std(dims+["time"])
        xds["nrmse"].attrs = {
            "label": "NRMSE",
        }

        # ACC / Cosine Similarity
        norm = {key: np.sqrt( (xds[key]**2).sum(dims) ) for key in [pkey, "truth"]}
        numerator = (xds[pkey] * xds["truth"]).sum(dims)
        denominator = norm[pkey]*norm["truth"]
        xds["acc"] =  numerator / denominator
        xds["acc"].attrs = {
            "description" : "Anomaly Correlation Coefficient, equivalent to Cosine Similarity since climatology is 0.",
            "label": "ACC",
        }
        return xds


    def calc_spectral_metrics(self, xds, pkey="prediction"):

        xsqg = XSQGTurb()

        # Get dataset with common time
        kds = xds.sel(time=self.time, method="nearest")
        ktrue = xsqg.calc_kespec1d(kds["truth"].load())
        kpred = xsqg.calc_kespec1d(kds[pkey].load())

        kerr = kpred - ktrue
        xds["ke_rel_err"] = kerr / np.abs(ktrue)
        xds["ke_rmse"] = np.sqrt( (kerr**2).mean("k1d") )
        xds["ke_nrmse"] = np.sqrt( ((kerr/ktrue.std("time"))**2).mean("k1d") )

        return xds
