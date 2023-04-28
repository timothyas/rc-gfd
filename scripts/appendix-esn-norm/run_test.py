

import os
import ddc

import sys
sys.path.append("../..")
from pw import submit_slurm_job

def main(name,
         reservoir_dimension,
         adjacency_norm="eig",
         input_method="dense"):

    config = "config_baseline.yaml"
    for random_state in range(10):
        output_directory = f"output-notspin-{name}/reservoir-{reservoir_dimension:04d}/random-state-{random_state:02d}"
        if not os.path.isfile(os.path.join(output_directory, "results.nc")):

            driver = ddc.CalValDriver(config=config, output_directory=output_directory)

            new_params = {
                "rc": {
                    "reservoir_dimension"   : reservoir_dimension,
                    "adjacency_norm"        : adjacency_norm,
                    "input_method"          : input_method,
                    "random_state"          : random_state,
                    },
                }

            driver.overwrite_params(new_params)
            driver.run_calval()


if __name__ == "__main__":

    jid = None
    rdlist = [4800]
    jkw = {
        "module_name"           : "run_test",
        "function_name"         : "main",
        "partition"             : "spot",
        }


    for reservoir_dimension in rdlist:

        # Run baseline
        submit_slurm_job(
            params=dict(
               name="eigA-scaleWin",
               reservoir_dimension=reservoir_dimension),
            **jkw)

        ## SVD for adjacency
        #submit_slurm_job(
        #    params=dict(
        #        name="svdA-scaleWin",
        #        adjacency_norm="svd",
        #        reservoir_dimension=reservoir_dimension),
        #    **jkw)

        # SVD for input matrix
        submit_slurm_job(
            params=dict(
                name="eigA-svdWin",
                input_method="dense-spectral",
                reservoir_dimension=reservoir_dimension),
            **jkw)

        # SVD for both
        submit_slurm_job(
            params=dict(
                name="svdA-svdWin",
                adjacency_norm="svd",
                input_method="dense-spectral",
                reservoir_dimension=reservoir_dimension),
            **jkw)

        ## sqrt(n_input) for input normalization
        #submit_slurm_job(
        #    params=dict(
        #        name="eigA-sqinWin",
        #        input_method="dense-sqin",
        #        reservoir_dimension=reservoir_dimension),
        #    **jkw)

        # sqrt(n_input) for input normalization + SVD A
        submit_slurm_job(
            params=dict(
                name="svdA-sqinWin",
                input_method="dense-sqin",
                adjacency_norm="svd",
                reservoir_dimension=reservoir_dimension),
            **jkw)
