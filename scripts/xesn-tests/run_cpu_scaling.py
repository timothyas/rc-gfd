
import os
from distributed import performance_report
from xesn import Driver

from distributed.diagnostics import MemorySampler
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../..")
from pw import submit_slurm_job

from run_slots_test import ms_to_xarray, get_client

def main(n_reservoir, n_local, n_workers, batch_size, n_nodes, n_years):

    outname = f"scaling-training-{n_reservoir:05d}res-{n_local:02d}loc-{n_workers:02d}work-{batch_size:05d}bs-{n_nodes:02d}node-{n_years:01d}year"
    experiment = outname.replace("scaling-","").replace("-", "_")
    driver = Driver(
        config="config-sqg.yaml",
        output_directory=outname,
    )
    driver.overwrite_config({
        "xdata": {
            "subsampling": {
                "time": {
                    "training": [None, n_years*103_680, None]
                }

            },
        },
        "lazyesn": {
             "n_reservoir": n_reservoir,
             "esn_chunks": {
                 "x": n_local,
                 "y": n_local,
                 "z": 2,
             },
        },
        "training": {
             "batch_size": batch_size,
        },
    })

    client = get_client(n_nodes=n_nodes, output_directory=driver.output_directory, workers_per_node=n_workers)

    ms = MemorySampler()
    report = os.path.join(driver.output_directory, "dask-report.html")
    with performance_report(report), ms.sample(experiment):
        driver.run_training()
    client.close()

    ms.plot(align=True)
    plt.savefig(f"{driver.output_directory}/memory_sampling.pdf", bbox_inches="tight")

    mds = ms_to_xarray(ms, name=experiment)
    mds.to_netcdf(f"{driver.output_directory}/memory_sampling.nc")


if __name__ == "__main__":

    one_week_mtu = 2016
    pdefault = {
        "n_reservoir": 2_000,
        "n_local": 8,
        "n_workers": 8,
        "batch_size": one_week_mtu*4, # 8064
        "n_nodes": 1,
        "n_years": 1,
    }

    for n_workers in [1, 2, 4, 8, 16]:
        params = pdefault.copy()
        params["n_workers"] = n_workers
        submit_slurm_job("run_cpu_scaling", "main", params, partition="compute")

    #for bs_factor in [1, 2, 4, 6, 8, 10]:
    #    params = pdefault.copy()
    #    params["batch_size"] = one_week_mtu * bs_factor
    #    submit_slurm_job("run_cpu_scaling", "main", params, partition="compute")
