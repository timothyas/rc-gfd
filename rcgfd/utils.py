from matplotlib.lines import Line2D
import xarray as xr

def global_legend(fig, labels, linewidth=10, color_start=0, **kwargs):

    lines = [Line2D([0], [0], color=f"C{i+color_start}", linewidth=linewidth) for i, _ in enumerate(labels)]

    if labels[-1].lower() == "persistence":
        lines.pop()
        lines.append(Line2D([0], [0], color=f"gray", linewidth=linewidth))

    loc = kwargs.pop("loc", "center")
    bbox_to_anchor = kwargs.pop("bbox_to_anchor", (0.5, -0.075))
    ncol = kwargs.pop("ncol", len(labels))
    frameon = kwargs.pop("frameon", False)
    fig.legend(
            lines,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            frameon=frameon,
            **kwargs
            )
    return


def replace_time(xds1, xds0):
    """Replace time in ``xds1`` using the final time stamp of ``xds0``

    Args:
        xds1, xds0 (:obj:`xarray.Dataset`): subsequent datasets (xds1 comes after xds0)

    Returns:
        xds1 (:obj:`xarray.Dataset`): with new time vector
    """

    new_time = xds1['time'].values + xds0['time'].isel(time=-1).values
    xds1 = xds1.rename({'time':'old_time'})
    xds1['time'] = xr.DataArray(new_time, xds1['old_time'].coords, xds1['old_time'].dims)
    xds1 = xds1.swap_dims({'old_time': 'time'}).drop('old_time')
    return xds1


def concatenate_time(dslist):
    """Concatenate datasets, creating new time vectors for the 2nd dataset and on, such that we have one continuous time dimension.

    Note:
        It is assumed that the final state of each dataset is equal to the initial state of the subsequent dataset. This routine chops off the initial condition from the 2nd dataset and on.

    Args:
        dslist (list of :obj:`xarray.Dataset`): to be concatenated along time

    Returns:
        xds (:obj:`xarray.Dataset`): with one continuous time
    """

    # Chop off initial condition from each
    dslist[1:] = [xds.isel(time=slice(1,None)) for xds in dslist[1:]]

    # Now modify the time vector for those datasets (note: list comprehension doesn't work here)
    for i in range(1,len(dslist)):
        dslist[i] = replace_time(dslist[i], dslist[i-1])

    # Concatenate
    xds = xr.concat(dslist, dim='time')

    # Attribute cleanup
    xds.attrs = dslist[0].attrs.copy()
    xds.attrs['trajectory_time'] = xds['time'].isel(time=-1).values.astype('<m8[s]').astype(int)
    xds.attrs['trajectory_steps'] = len(xds['time']) - 1

    return xds
