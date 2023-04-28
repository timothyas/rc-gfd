import sys
sys.path.append("../..")

from rcgfd import SQGTurbGenerator

if __name__ == '__main__':

    Nx = 64
    n_years = 5 # repeat 5 times to make 25 years...

    pickup_zstore = None
    for i in range(5):

        zstore = f"sqg.{i}.{Nx:03d}n.{n_years:03d}years.02z.01y.01x.zarr"

        gen = SQGTurbGenerator(zstore=zstore,
                               Nx=Nx,
                               delta_t=int(1200 / 4),
                               trajectory_time=n_years*360*24*3600,
                               logfile=f'stdout.{i}.{n_years}years.{Nx:03d}n.log',
                               chunksize={'time':None,
                                          'z':2,
                                          'y':1,
                                          'x':1},
                               threads=5)

        gen(pickup_zstore=pickup_zstore)

        pickup_zstore = zstore
