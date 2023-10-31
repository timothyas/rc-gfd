
import os
from dask_jobqueue import SLURMCluster
from distributed import Client, performance_report, LocalCluster
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


def get_client(n_nodes, output_directory, workers_per_node=8):

    if n_nodes == 1:
        cluster = LocalCluster(
            local_directory=os.path.join(output_directory, "worker-logs"),
            n_workers=workers_per_node,
        )
    else:
        cluster = SLURMCluster(
            log_directory=os.path.join(output_directory, "worker-logs"),
            processes=workers_per_node,
            queue=os.getenv("SLURM_JOB_PARTITION"),
            job_script_prologue=[
                "source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh",
                "conda activate xesn",
            ],
        )
        cluster.scale(jobs=n_nodes)

    client = Client(cluster)

    if n_nodes > 1:
        client.wait_for_workers(n_nodes*workers_per_node)
    return client


def main(branch, n_nodes, n_years):

    outname = f"output-training-{branch}-{n_years}year-{n_nodes}node"
    experiment = outname[7:].replace("-", "_")
    driver = Driver(
        config="config-sqg.yaml",
        output_directory=outname,
    )
    driver.overwrite_config({"xdata": {"subsampling": {"time": {"training": [None, n_years*103_680, None]}}}})

    client = get_client(n_nodes=n_nodes, output_directory=driver.output_directory)

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
    params = {
        "branch": "main",
        "n_nodes": 1,
        "n_years": 5,
    }
    submit_slurm_job("run_training", "main", params, partition="compute")
