import numpy as np
import xarray as xr

from ddc import YAMLParser

class RCDataset():

    n_sub = None
    cost_terms = None
    n_samples = 50

    chunks = None

    def __init__(self, **kwargs):
        for key,val in kwargs.items():
            try:
                getattr(self, key)
            except:
                raise
            setattr(self, key, val)

        self.chunks = {"x":16, "y":16} if self.chunks is None else self.chunks
        self.n_sub = (self.n_sub,) if not isinstance(self.n_sub, (list, tuple)) else self.n_sub
        self.cost_terms = (self.cost_terms,) if not isinstance(self.cost_terms, (list, tuple)) else self.cost_terms


    def __call__(self):

        dslist=[]
        for n_sub in self.n_sub:
            dslist2 = []
            for ct in self.cost_terms:
                dslist2.append( self.open_single_dataset(ct, n_sub) )
            dslist.append(xr.concat(dslist2, dim='experiment'))

        xds = xr.concat(dslist, dim='n_sub')
        xds = xds.squeeze()

        xds.attrs['keep_spinup'] = 'False'
        xds.attrs['spinup_steps_predict'] = 0

        xds["truth"] = xds["truth"].isel(experiment=0, n_sub=0)
        xds = self.calc_metrics(xds)
        return xds


    def open_single_dataset(self, this_cost, n_sub):


        main_dir = "cost-"
        main_dir += "-".join(f"{k}{v:1.1e}" for k,v in this_cost.items())

        dt0 = 300
        delta_t = n_sub * dt0
        out_dir = f'/contrib/Tim.Smith/qgrc-teachers/sqg/resolution/rc-temporal-sampling-064n/{main_dir}/'
        out_dir += f'validation-{delta_t:04d}dt-{self.n_samples:03d}samples/'
        fname = out_dir + "results.zarr"

        xds = xr.open_zarr(fname, chunks=self.chunks)

        # Determine if we need to un-normalize the field
        yp = YAMLParser()
        c = yp.read(f"{out_dir}/config-lazy.yaml")
        sd = c["preprocessing"]["norm_factor"]
        if sd is not None:
            with xr.set_options(keep_attrs=True):
                for key in ["truth", "prediction"]:
                    xds[key] = xds[key]*sd

        experiment = main_dir.replace("cost-","")
        xds = xds.expand_dims({
            'n_sub': [n_sub],
            'experiment': [experiment]
        })
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
