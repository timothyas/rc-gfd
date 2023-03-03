import numpy as np
import xarray as xr

from ddc import YAMLParser

class Dataset():

    n_sub       = None
    n_samples   = 50

    chunks      = None
    dims        = ("n_sub",)

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

        xds = xds.squeeze()
        xds.attrs['keep_spinup'] = 'False'
        xds.attrs['spinup_steps_predict'] = 0

        for key in self.dims:
            if key in xds["truth"].dims:
                xds["truth"] = xds["truth"].isel({key: 0})

        xds = self.calc_metrics(xds)
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


    def calc_metrics(self, xds):

        xds["error"] = xds["prediction"] - xds["truth"]
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
        norm = {key: np.sqrt( (xds[key]**2).sum(dims) ) for key in ["prediction", "truth"]}
        numerator = (xds["prediction"] * xds["truth"]).sum(dims)
        denominator = norm["prediction"]*norm["truth"]
        xds["acc"] =  numerator / denominator
        xds["acc"].attrs = {
            "description" : "Anomaly Correlation Coefficient, equivalent to Cosine Similarity since climatology is 0.",
            "label": "ACC",
        }
        return xds
