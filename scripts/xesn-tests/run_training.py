
import os
from dask_jobqueue import SLURMCluster
from distributed import Client, performance_report
from xesn import Driver

from distributed.diagnostics import MemorySampler
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import sys
sys.path.insert(0, "../..")
from pw import submit_slurm_job

def ms_to_xarray(ms, name):

    table = ms.to_pandas()
    time = [np.datetime64(x) for x in table.index]
    delta_t = time - time[0]
    vals = [x for x in table[name].values]
    xda = xr.DataArray(vals, coords={'delta_t':delta_t}, dims=('delta_t',), name=name)
    return xda.to_dataset()

def main(branch):

    workers_per_node = 8
    n_nodes = 2

    driver = Driver(
        config="config-sqg.yaml",
        output_directory=f"output-training-{branch}",
    )
    cluster = SLURMCluster(
        log_directory=os.path.join(driver.output_directory, "worker-logs"),
        processes=workers_per_node,
        queue=os.getenv("SLURM_JOB_PARTITION"),
        job_script_prologue=[
            "source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh",
            "conda activate xesn",
        ],
    )
    cluster.scale(jobs=n_nodes)
    client = Client(cluster)

    ms = MemorySampler()
    report = os.path.join(driver.output_directory, "dask-report.html")
    with performance_report(report), ms.sample(branch):
        driver.run_training()
    client.close()

    ms.plot(align=True)
    plt.savefig(f"{driver.output_directory}/memory_sampling.pdf", bbox_to_inches="tight")

    mds = ms_to_xarray(ms, name=branch)
    mds.to_netcdf(f"{driver.output_directory}/memory_sampling.nc")


if __name__ == "__main__":
    params = {
        "branch": "main",
    }
    submit_slurm_job("run_training", "main", params, partition="compute")

