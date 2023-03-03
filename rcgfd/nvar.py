import numpy as np
import xarray as xr

from ddc import YAMLParser
from .io import Dataset

class NVARDataset(Dataset):

    n_lag       = None
    n_overlap   = 1

    dims        = ("n_sub", "n_lag")

    def __init__(self, **kwargs):

        self.n_lag = kwargs.pop("n_lag", None)
        self.n_lag = (self.n_lag) if not isinstance(self.n_lag, (list, tuple)) else self.n_lag
        super().__init__(**kwargs)


    def __call__(self):

        dslist=[]
        for n_sub in self.n_sub:
            dslist2 = []
            for n_lag in self.n_lag:
                dslist2.append( self.open_single_dataset(n_lag, n_sub) )
            dslist.append(xr.concat(dslist2, dim='n_lag'))
        xds = xr.concat(dslist, dim='n_sub')

        xds = self.postprocess_dataset(xds)
        return xds


    def get_results_path(self, n_lag, n_sub):

        dt0 = 300
        delta_t = n_sub * dt0
        main_dir = f"validation-{self.n_samples:03d}samples/{delta_t:04d}dt-lag{n_lag:02d}-nb{self.n_overlap:02d}/"
        fname = f"/contrib/Tim.Smith/qgrc-teachers/sqg/resolution/nvar-temporal-sampling-064n/{main_dir}/results.zarr"
        return main_dir, fname


    def open_single_dataset(self, n_lag, n_sub):


        main_dir, fname = self.get_results_path(n_lag, n_sub)

        try:
            xds = xr.open_zarr(fname, chunks=self.chunks)
        except:
            xds = xr.concat(
                    [xr.open_zarr(fname.replace("results", f"results.{i:03d}"), chunks=self.chunks) for i in range(self.n_samples)],
                    dim="sample",
                    coords="minimal")

        xds = xds.expand_dims({
            'n_sub': [n_sub],
            'n_lag': [n_lag]
        })
        return xds
