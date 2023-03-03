import numpy as np
import xarray as xr

from .io import Dataset

class RCDataset(Dataset):

    cost_terms  = None
    dims        = ("n_sub", "experiment")

    def __init__(self, **kwargs):
        self.cost_terms = kwargs.pop("cost_terms", None)
        self.cost_terms = (self.cost_terms,) if not isinstance(self.cost_terms, (list, tuple)) else self.cost_terms
        super().__init__(**kwargs)


    def __call__(self):

        dslist=[]
        for n_sub in self.n_sub:
            dslist2 = []
            for ct in self.cost_terms:
                dslist2.append( self.open_single_dataset(ct, n_sub) )
            dslist.append(xr.concat(dslist2, dim='experiment'))

        xds = xr.concat(dslist, dim='n_sub')
        xds = self.postprocess_dataset(xds)
        return xds


    def get_results_path(self, cost, n_sub):

        main_dir = "cost-"
        main_dir += "-".join(f"{k}{v:1.1e}" for k,v in cost.items())

        dt0 = 300
        delta_t = n_sub * dt0
        out_dir = f'/contrib/Tim.Smith/qgrc-teachers/sqg/resolution/rc-temporal-sampling-064n/{main_dir}/'
        out_dir += f'validation-{delta_t:04d}dt-{self.n_samples:03d}samples/'
        fname = out_dir + "results.zarr"
        return main_dir, fname


    def open_single_dataset(self, this_cost, n_sub):


        main_dir, fname = self.get_results_path(this_cost, n_sub)

        try:
            xds = xr.open_zarr(fname, chunks=self.chunks)
        except:
            xds = xr.concat(
                    [xr.open_zarr(fname.replace("results", f"results.{i:03d}"), chunks=self.chunks) for i in range(self.n_samples)],
                    dim="sample",
                    coords="minimal")

        xds = self.renormalize_dataset(xds, fname)
        experiment = main_dir.replace("cost-","")
        xds = xds.expand_dims({
            'n_sub': [n_sub],
            'experiment': [experiment]
        })
        return xds
