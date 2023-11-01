
import os
from distributed import performance_report
from xesn import Driver

from distributed.diagnostics import MemorySampler

import sys
sys.path.insert(0, "../..")
from pw import submit_slurm_job
from rcgfd import ScalingTest

from run_slots_test import get_client

def main(n_reservoir, n_local, n_workers, batch_size, n_nodes, n_years, sample):

    st = ScalingTest(
        n_reservoir=n_reservoir,
        n_local=n_local,
        n_workers=n_workers,
        batch_size=batch_size,
        n_nodes=n_nodes,
        n_years=n_years,
        sample=sample
    )

    driver = Driver(
        config="config-sqg.yaml",
        output_directory=st.output_directory,
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
    with performance_report(report), ms.sample("training"):
        driver.run_training()
    client.close()

    st.store_memory_sampling(ms, "training")


if __name__ == "__main__":

    pdefault = {
        "n_reservoir": 2_000,
        "n_local": 8,
        "n_workers": 8,
        "batch_size": 8*288, # 1 day
        "n_nodes": 1,
        "n_years": 1,
        "sample": 0,
    }

    #for bs_factor in [1, 2, 8, 16, 32, 64]:
    #    for sample in range(3):
    #        params = pdefault.copy()
    #        params["batch_size"] = 288 * bs_factor
    #        params["sample"] = sample
    #        st = ScalingTest(**params)
    #        if not os.path.isfile(st.fname):
    #            submit_slurm_job("run_cpu_scaling", "main", params, partition="compute")

    for n_workers in [1, 2, 4, 8, 16, 32]:
        for sample in range(3):
            params = pdefault.copy()
            params["n_workers"] = n_workers
            params["sample"] = sample
            st = ScalingTest(**params)
            if not os.path.isfile(st.fname):
                submit_slurm_job("run_cpu_scaling", "main", params, partition="compute")

    #for n_reservoir in [2_000, 4_000, 8_000]:
    #    params = pdefault.copy()
    #    params["n_reservoir"] = n_reservoir
    #    submit_slurm_job("run_cpu_scaling", "main", params, partition="compute")

    #for n_local in [2, 4, 8, 16]:
    #    params = pdefault.copy()
    #    params["n_local"] = n_local
    #    submit_slurm_job("run_cpu_scaling", "main", params, partition="compute")
