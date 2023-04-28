"""Concatenate the zarr stores"""

import xarray as xr
import zarr

import sys
sys.path.append("../..")
from rcgfd import concatenate_time


if __name__ == "__main__":

    fname = lambda i : f"sqg.{i}.064n.005years.02z.02y.02x.zarr"
    dslist = []

    for i in range(5):

        xds = xr.open_zarr(fname(i))
        xds = xds['theta'].to_dataset()
        dslist.append(xds)

    xds = concatenate_time(dslist)

    # just in case...
    xds['theta'].encoding = {}
    xds['theta'] = xds['theta'].chunk({
        'time'  : 103_680,
        'z'     : 2,
        'y'     : 2,
        'x'     : 2})

    path = f"sqg.theta.0300dt.064n.100kt.02z.02y.02x.zarr"
    store = zarr.NestedDirectoryStore(path=path)
    xds.to_zarr(store=store)
