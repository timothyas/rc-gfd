
import numpy as np
import xarray as xr
import scipy.fft as sfft
import dask.array.fft as dfft

class XSQGTurb():

    N = 64

    H = 1e4
    Lr= 1e6
    L = 20*1e6
    nsq = 1e-4
    f = 1e-4
    g = 9.8
    U = 30
    theta0 = 300
    symmetric = True

    @property
    def x(self):
        """This is consistent with my generator, but not with SQGturb ... just due
        to linspace vs arange
        """
        x = np.linspace(0, self.L, self.N)
        x = x/1e3
        return xr.DataArray(x,
                            coords={'x':x},
                            dims=('x'),
                            attrs={'units': 'km',
                                   'description': 'zonal (x) coordinate'})
    @property
    def y(self):
        y = np.linspace(0, self.L, self.N)
        y = y/1e3
        return xr.DataArray(y,
                            coords={'y':y},
                            dims=('y'),
                            attrs={'units': 'km',
                                   'description': 'meridional (y) coordinate'})
    @property
    def z(self):
        z = np.array([0, self.H])
        z = z/1e3
        return xr.DataArray(z,
                            coords={'z':z},
                            dims=('z'),
                            attrs={'units':'km',
                                   'description': 'vertical (z) coordinate'})

    @property
    def kx(self):
        kx = np.abs((self.N*sfft.fftfreq(self.N)))[:self.N//2+1]
        return xr.DataArray(kx,
                            coords={'kx': kx},
                            dims=('kx',),
                            attrs={'units': '',
                                   'description':'Nondimensional wavenumber in x-direction'})
    @property
    def ky(self):
        ky = self.N*sfft.fftfreq(self.N)
        return xr.DataArray(ky,
                          coords={'ky': ky},
                          dims=('ky',),
                          attrs={'units': '',
                                 'description':'Nondimensional wavenumber in y-direction'})

    @property
    def kmag(self):
        return np.sqrt(self.kx**2 + self.ky**2)

    @property
    def kcutoff(self):
        return np.pi * self.N / self.L

    @property
    def l(self):
        return 2 * np.pi / self.L

    @property
    def mubar(self):
        return self.l * np.sqrt(self.nsq) * self.H / self.f

    @property
    def mu(self):
        mu = np.sqrt(self.kmag) * np.sqrt(self.nsq) * self.H / self.f
        mu = mu.clip(np.finfo(mu).eps)
        return mu.astype(np.float64)


    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


    def calc_pvbar(self):

        pvbar = -(self.mubar*self.U) / (self.l * self.H) * np.cos(self.l*self.y)

        if self.symmetric:
            pvbar = pvbar * 0.5 * np.cosh(.5*self.mubar) / np.sinh(0.5*self.mubar)

        else:
            pvbar = pvbar / np.sinh(mubar)

        pvbar = pvbar.broadcast_like(self.x).broadcast_like(self.z).transpose('x','y','z')

        if not self.symmetric:
            pvbar = pvbar.copy()
            pvbar[...,-1] = pvbar[...,-1]*np.cosh(self.mubar)

        return pvbar

    def theta2pv(self, arr):
        return arr * self.f * self.theta0 / self.g


    def pv2theta(self, arr):
        return arr * self.g / self.f / self.theta0


    def rfft2(self, arr, **kwargs):
        """Apply along (x,y) axes
        """
        dims = arr.dims
        axes = tuple(dims.index(x) for x in ('y','x'))
        rfft2 = dfft.rfft2 if hasattr(arr.data, "chunks") else sfft.rfft2
        spec = rfft2(arr.data, axes=axes, **kwargs)

        kdims = tuple(x if x not in ('y','x') else "k"+x for x in dims)

        coords = {}
        for key in kdims:
            if key in dims:
                coords[key] = arr[key]
            else:
                coords[key] = getattr(self, key)

        spec = xr.DataArray(spec,
                            coords=coords,
                            dims=kdims,
                            attrs={'description': f'rfft2 of {arr.name}'})
        return spec


    def irfft2(self, arr, **kwargs):
        """Apply along (x,y) axes
        """
        dims = arr.dims
        axes = tuple(dims.index(x) for x in ('ky','kx'))
        irfft2 = dfft.irfft2 if hasattr(arr.data, "chunks") else sfft.irfft2
        grid_arr = irfft2(arr.data, axes=axes, **kwargs)

        gdims = tuple(x if x not in ('ky','kx') else x.replace('k','') for x in dims)

        coords = {}
        for key in gdims:
            if key in dims:
                coords[key] = arr[key]
            else:
                coords[key] = getattr(self, key)

        grid_arr = xr.DataArray(grid_arr,
                                coords=coords,
                                dims=gdims,
                                attrs={'description': f'irfft2 of {arr.name}'})
        return grid_arr


    def invert(self, pvspec):
        """invert (rfft2 of) potential vorticity for streamfunction spectra
        """
        sh = np.sinh(self.mu).astype(np.float32)
        th = np.tanh(self.mu).astype(np.float32)
        Hmu = self.H / self.mu

        psispec0 = Hmu * (pvspec.isel(z=1)/sh - pvspec.isel(z=0)/th)
        psispec1 = Hmu * (pvspec.isel(z=1)/th - pvspec.isel(z=0)/sh)
        return xr.concat([psispec0, psispec1], dim='z')


    def derivative(self, arr, dim, dealias=True):
        """Compute derivative of array using FFT

        Args:
            arr (array_like): array in grid or spectral space
            dim (str): dimension to take derivative along, either "x" or "y"
            dealias (bool, optional): use 2/3 rule to dealias (pad/truncate) result

        Returns:
            d_arr (array_like): derivative of array
        """

        assert dim in ("x", "y")
        k = self.kx if dim == "x" else self.ky
        spec = self.rfft2(arr) if "kx" not in arr.dims else arr
        return self.irfft2(spec * k * 1.j)


    def calc_kespec1d(self, theta, dimensional_wavenumbers=True):
        """Compute 1D KE spectrum
        """
        pv = theta * self.g/self.f/self.theta0
        pvgrd = pv - self.calc_pvbar()

        pvspec = self.rfft2(pv)
        psispec = self.invert(pvspec)
        psispec = psispec / (self.N * np.sqrt(2))

        kespec = self.kmag * (psispec * np.conjugate(psispec)).real

        # make 1D version by dropping k,z, create k1d
        tmp = kespec.isel(kx=0,ky=0,z=0)
        kespec1d = xr.zeros_like(tmp)

        kmag = self.kmag # compute this once
        kmax = len(self.kx)

        # wavenumbers
        k1d = np.arange(kmax)
        if dimensional_wavenumbers:
            k1d = k1d / self.L * 2 * np.pi * 1e3
            attrs = {'units': 'km', 'description': '1D wavenumber'}
        else:
            attrs = {'units': '', 'description': 'nondimensional 1D wavenumber'}

        k1d = xr.DataArray(k1d, coords={'k1d':k1d}, dims=('k1d',), attrs=attrs)
        kespec1d = kespec1d.expand_dims({'k1d': k1d}).copy()

        for ikx in range(self.N//2+1):
            for iky in range(self.N):

                this_k = int(kmag.isel(kx=ikx, ky=iky))
                if this_k < kmax:
                    result = kespec.isel(kx=ikx,ky=iky).mean('z')
                    kespec1d[{'k1d':this_k}] += xr.where(this_k < kmax, result, 0.)


        kespec1d.name = "KE Density"
        return kespec1d
