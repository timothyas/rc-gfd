"""Script to generate a long time series of data and save to zarr
Note: Nice brief description of the model in 10.1175/MWR-D-20-0290.1
"""

import os
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import zarr
from dask.distributed import Client
from pyfftw.interfaces import numpy_fft

import ddc

class SQGTurbGenerator():

    # SQG Parameters
    Nx              = 64                # Number of grid cells, x and y equal
    Ny              = None              # Set based on rfft2, usually truncated / less than Nx
    Nz              = 2                 # Number of vertical levels
    delta_t         = None              # Time step, in seconds
    diff_efold      = None              # Timescale for hyperdiffusion at smallest resolved scale (s)

    diff_order      = 8                 # Order of hyperdiffusion
    dealias         = True              # dealiased with 2/3 rule?
    dek             = 0                 # Only applied at surface if symmetric=False
    nsq             = 1.e-4             # Brunt-Vaisala frequency
    f               = 1.e-4             # Coriolis parameter
    g               = 9.8               # Gravity (m/s^2)
    theta0          = 300               # (K)

    H               = 1.e4              # Lid height (m)
    U               = 30                # Jet speed (m/s)
    tdiab           = 10.*86400         # timescale for linear thermal relaxation to equilibrium (s)
    symmetric       = True              # if False, asymmetric equilibrium jet with zero wind at surface

    # Computed quantities
    r               = None              # Ekamn damping coefficient
    Lr              = None              # Rossby radius
    L               = None
    scale_fact      = None              # Used to scale PV to temprerature units

    # Zarr stuff
    chunksize       = {'time' : None,
                       'z' : 2,
                       'y' : 2,
                       'x' : 2}

    # Extra
    tstart          = 0
    spinup_time     = 360 * 24 * 3600   # 360 days, in seconds
    trajectory_time = 720 * 24 * 3600   # 720 days, in seconds
    pv0_random_seed = 0
    time_units      = 's'
    precision       = 'single'
    threads         = int(os.getenv('OMP_NUM_THREADS', '30'))
    logfile         = 'stdout.log'

    @property
    def nbytes_spinup(self):
        n = self.spinup_steps * self.Nx * self.Nx * self.Nz
        b = 4 if self.precision == 'single' else 8
        return n*b


    @property
    def nbytes_trajectory(self):
        n = self.trajectory_steps * self.Nx * self.Nx * self.Nz
        b = 4 if self.precision == 'single' else 8
        return n*b


    def __init__(self, zstore, **kwargs):

        for key, val in kwargs.items():
            if key == "n_x":
                raise KeyError(f"Unrecognized option {key}.")
            setattr(self, key, val)

        self._set_timespace()

        self.zstore = zstore
        if self.precision == 'single':
            self.dtype = np.dtype('float32')
        elif self.precision == 'double':
            self.dtype = np.dtype('float64')
        else:
            raise ValueError(f"SQGTurbGenerator.__init__: precision must be 'single' or 'double', got {self.precision}")

        # Computed quantities
        self.r = self.dek * self.nsq / self.f
        self.Lr = np.sqrt(self.nsq) * self.H / self.f
        self.L = 20. * self.Lr
        self.scale_fact = self.f * self.theta0 / self.g

        self.x = np.linspace(0, self.L, self.Nx, dtype=self.dtype)
        self.y = np.linspace(0, self.L, self.Nx, dtype=self.dtype)
        self.z = np.linspace(0, self.H, self.Nz, dtype=self.dtype)

        self.localtime = ddc.Timer(filename=self.logfile)
        self.walltime = ddc.Timer(filename=self.logfile)



    def __call__(self, pickup_zstore=None):

        self.walltime.start("Starting SQGTurb Data Generation")

        # Create and spinup
        dataobj = self.create_object()

        if pickup_zstore is None:

            self.localtime.start("Spinup")
            dataobj = self.spinup(dataobj)
            fig, ax = self.plot_pv(dataobj, cmap='plasma')
            fig.savefig("spunup.jpg", bbox_inches='tight', dpi=300)
            self.localtime.stop()

        else:

            self.localtime.start(f"Picking up from {pickup_zstore}")
            xds = xr.open_zarr(pickup_zstore)

            x0 = numpy_fft.rfft2(xds['q'].isel(time=-1).values).ravel()
            del xds

            dataobj.x0 = x0
            self.localtime.stop()

        self.localtime.start("Generating trajectory")
        dataobj.generate(n_steps=self.trajectory_steps)
        self.localtime.stop()

        self.localtime.start("Converting to xarray")
        xds = self.dataobj_to_xarray(dataobj)
        self.localtime.stop()

        self.localtime.start("Chunking and storing")
        self.xds_to_zarr(xds)
        self.localtime.stop()

        self.walltime.stop("Total Walltime")


    def _set_timespace(self):
        # a dictionary with possible combos
        # see https://github.com/jswhit/sqgturb/blob/master/sqg_run.py
        combos = {32:
                    {'delta_t'      : 1800,         # 30 min
                     'diff_efold'   : 86400*2,      # 2 days
                    },
                  64:
                    {'delta_t'      : 1200,         # 20 min
                     'diff_efold'   : 86400,        # 1 day
                    },
                  96:
                    {'delta_t'      : 900,          # 15 min
                     'diff_efold'   : 86400 / 3,
                    },
                  128:
                    {'delta_t'      : 600,          # 10 min
                     'diff_efold'   : 86400 / 3,
                    },
                  192:
                    {'delta_t'      : 300,          # 5 min
                     'diff_efold'   : 86400 / 8,
                    },
                  256:
                    {'delta_t'      : 180,          # 3 min
                     'diff_efold'   : 86400 / 16
                    },
                  512:
                    {'delta_t'      : 90,          # 1.5 min
                     'diff_efold'   : 86400 / 48
                     }
                 }
        try:
            assert self.Nx in list(combos.keys())
        except AssertionError:
            raise ValueError(f"Got unexpected size Nx={Nx}, reset to one of {list(combos.keys())}")

        self.delta_t = combos[self.Nx]['delta_t'] if self.delta_t is None else self.delta_t
        self.diff_efold = combos[self.Nx]['diff_efold'] if self.diff_efold is None else self.diff_efold

        self.spinup_steps = int(self.spinup_time / self.delta_t)
        self.trajectory_steps = int(self.trajectory_time / self.delta_t)


    def print_log(self, text):
        with open(self.logfile, 'a') as file:
            with redirect_stdout(file):
                print(text)


    def random_initial_condition(self, noise_scale=100., amp=2000., n_exp=20):

        rs = np.random.RandomState(seed=self.pv0_random_seed)
        pv = rs.normal(loc=0., scale=noise_scale, size=(self.Nz, self.Nx, self.Nx))
        pv = pv.astype(self.dtype)

        x_g = np.linspace(0, 2.*np.pi, self.Nx, dtype=self.dtype)
        y_g = np.linspace(0, 2.*np.pi, self.Nx, dtype=self.dtype)
        x_g, y_g = np.meshgrid(x_g, y_g)
        pv[1] = pv[1] + amp * np.sin(x_g/2)**(2*n_exp) * np.sin(y_g)**n_exp

        pv = pv - np.nanmean(pv, axis=0)
        return pv


    def create_object(self, pv0=None):
        """
        Args:
            pv0 (array_like): 3D PV initial condition

        Returns:
            dataobj (DataSQGturb): initialized object
        """

        pv0 = self.random_initial_condition() if pv0 is None else pv0

        kw = {}
        for key in ['nsq', 'f', 'L', 'H', 'U', 'r', 'tdiab', 'diff_order',
                    'diff_efold', 'symmetric', 'delta_t', 'dealias',
                    'threads', 'precision', 'tstart']:
            kw[key] = getattr(self, key)

        self.print_log(" --- Creating Object with parameters:")
        for key, val in kw.items():
            self.print_log(f"{key:<24s}: {val}")

        dataobj = ddc.DataSQGturb(pv0, **kw)
        self.Ny = dataobj.Ny
        return dataobj


    def spinup(self, dataobj, x0=None):
        """
        Returns:
            x0 (array_like): new initial conditions, from spunup state
        """

        dataobj.generate(x0=x0, t_final=self.spinup_time)

        dataobj.x0 = dataobj.values[:,-1]
        return dataobj.slice_time([-1])


    def plot_pv(self, dataobj, index=-1, level=1, **kwargs):
        """A quick plotting method ..."""

        plotme = dataobj.map1dto2d_ifft2(dataobj.values[:,index])
        plotme = plotme[level]

        fig, ax = plt.subplots(constrained_layout=True)
        ax.pcolormesh(self.x, self.y, plotme, **kwargs)

        return fig, ax


    def map_1dtime_to_4d(self, array):
        """This should get wrapped into DataSQGturb - basically does this ``irfft2``
        for arrays varying in time
        """

        n_time = array.shape[-1]
        reshaped = array.T.reshape((n_time, self.Nz, self.Nx, self.Ny))
        return numpy_fft.irfft2(reshaped, threads=self.threads)


    def index_to_time(self, ind):
        """map to :obj:`np.timedelta64`"""
        if np.array(ind).shape == ():
            return np.timedelta64(ind*self.delta_t, self.time_units)
        else:
            return np.array([np.timedelta64(x*self.delta_t, self.time_units) for x in ind])


    def pv_to_theta(self, pv):
        return pv * self.scale_fact


    def dataobj_to_xarray(self, dataobj):

        arr4d = self.map_1dtime_to_4d(dataobj.values)

        coords = {'time' : self.index_to_time(np.arange(arr4d.shape[0])),
                  'z': self.z / 1000,
                  'y': self.y / 1000,
                  'x': self.x / 1000}
        dims = tuple(coords.keys())
        xds = xr.DataArray(arr4d,
                           coords=coords,
                           dims=dims,
                           name='q',
                           attrs={'units': 'g/f',
                                  'description': 'potential vorticity'})

        xds = xds.to_dataset()
        theta = arr4d * self.f
        xds['theta'] = xr.DataArray(self.pv_to_theta(arr4d),
                                    coords=coords,
                                    dims=dims,
                                    attrs={'units' : 'K',
                                           'description' : f'temperature deviation from {self.theta0} K'})


        for key in ['nsq', 'f', 'L', 'H', 'U', 'r', 'tdiab', 'diff_order',
                    'diff_efold', 'symmetric', 'delta_t', 'dealias',
                    'threads', 'precision', 'tstart',
                    'Lr', 'theta0', 'scale_fact',
                    'spinup_time', 'trajectory_time',
                    'spinup_steps', 'trajectory_steps']:
            val = getattr(self, key)
            if isinstance(val, (bool, dict, type)) or val is None:
                xds.attrs[key] = str(val)
            else:
                xds.attrs[key] = val

        # Note: adding time "units" attribute causes errors, for some reason...
        # Maybe because it's already a timedelta object
        # Note: saving with km rather than m ...
        xds['z'].attrs['units'] = 'km'
        xds['y'].attrs['units'] = 'km'
        xds['x'].attrs['units'] = 'km'
        self.print_log("Created dataset:")
        self.print_log(str(xds))
        return xds

    def xds_to_zarr(self, xds):

        for key in xds.data_vars:
            xds[key] = xds[key].chunk(self.chunksize)

        self.print_log("Rechunked dataset:")
        self.print_log(str(xds))

        store = zarr.NestedDirectoryStore(path=self.zstore)
        xds.to_zarr(store=store)
        self.print_log(f"saved to: {self.zstore}")
