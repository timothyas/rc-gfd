
import os
from dask_jobqueue import SLURMCluster
from distributed import Client
from xesn import Driver

import sys
sys.path.insert(0, "../..")
from pw import submit_slurm_job

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
    )
    cluster.scale(jobs=n_nodes)

    driver.run_training()
    client.close()

if __name__ == "__main__":
    params = {
        "branch": "main",
    }
    submit_slurm_job("run_training", "main", params, partition="compute")

