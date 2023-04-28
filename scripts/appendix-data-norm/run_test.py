

import os
import ddc

import sys
sys.path.append("../..")
from pw import submit_slurm_job

def main(name,
         distribution="uniform",
         adjacency_norm="eig",
         input_method="dense-spectral",
         spinup_steps_train=0,
         readout_method="linear",
         misfit_type="sum",
         time_dimension_train=None,
         norm_factor=None,
         norm_subtract=None,
         noise_amplitude=None,
         add_noise_test=False):

    config = "config_bigbounds.yaml" if "bigbounds" in name else "config_baseline.yaml"
    for random_state in range(10):
        output_directory = f"output-{name}/random-state-{random_state:02d}"
        if not os.path.isfile(os.path.join(output_directory, "results.nc")):

            driver = ddc.CalValDriver(config=config, output_directory=output_directory)

            new_params = {
                "rc": {
                    "adjacency_distribution": distribution,
                    "input_distribution"    : distribution,
                    "adjacency_norm"        : adjacency_norm,
                    "input_method"          : input_method,
                    "spinup_steps_train"    : spinup_steps_train,
                    "random_state"          : random_state,
                    "readout_method"        : readout_method,
                    "misfit_type"           : misfit_type,
                    },
                "data": {},
                "preprocessing": {},
                }

            if norm_factor is not None:
                new_params["preprocessing"]["norm_factor"] = norm_factor

            if norm_subtract is not None:
                new_params["preprocessing"]["norm_subtract"] = norm_subtract

            if add_noise_test:
                new_params["preprocessing"]["add_noise_test"] = True
                new_params["preprocessing"]["noise_amplitude"] = noise_amplitude

            if noise_amplitude is not None and not add_noise_test:
                new_params["data"]["noise_amplitude"] = noise_amplitude
                new_params["data"]["add_noise"] = True

            if time_dimension_train is not None:
                new_params["data"]["time_dimension_train"] = time_dimension_train


            if "ortho" in distribution:
                new_params["rc"]["input_distribution"] = "uniform"
                new_params["rc"]["sparse_adj_matrix"] = False

            driver.overwrite_params(new_params)
            driver.run_calval()


if __name__ == "__main__":

    jid = None
    jkw = {
        "module_name"           : "run_test",
        "function_name"         : "main",
        "partition"             : "spot",
        }

    ## Run baseline
    #submit_slurm_job(
    #    params=dict(name="baseline"),
    #    **jkw)

    ## Test spinup steps
    #submit_slurm_job(
    #    params=dict(name="tspin", spinup_steps_train=500),
    #    **jkw)

    ## Test distribution:
    #submit_slurm_job(
    #    params=dict(name="normal", distribution="normal"),
    #    **jkw)

    ## Test normalization by svd
    #submit_slurm_job(
    #    params=dict(name="svd", adjacency_norm="svd"),
    #    **jkw)

    ## both of the above
    #submit_slurm_job(
    #    params=dict(name="normal-svd", distribution="normal", adjacency_norm="svd"),
    #    **jkw)

    ## all of the above
    #submit_slurm_job(
    #    params=dict(name="normal-svd-tspin", distribution="normal", adjacency_norm="svd", spinup_steps_train=500),
    #    **jkw)

    ## For reference, bigbounds and nothing else
    #submit_slurm_job(
    #        params=dict(name="bigbounds"),
    #        **jkw)

    ## data normalization
    #for prefix in ["bigbounds-"]:

    #    # Test with data normalization by scalar
    #    submit_slurm_job(
    #            params=dict(name=prefix+"normstd", norm_factor=3.67, norm_subtract=2.42),
    #            **jkw)

    #    # Test with data normalization by scalar, and SVD
    #    submit_slurm_job(
    #            params=dict(name=prefix+"svd-normstd", adjacency_norm="svd", norm_factor=3.67, norm_subtract=2.42),
    #            **jkw)

    #    # test maxmin normalization from Platt et al
    #    submit_slurm_job(
    #            params=dict(name=prefix+"normmaxmin", norm_factor=21.46, norm_subtract=2.42),
    #            **jkw)

    #    # maxmin normalization with svd
    #    submit_slurm_job(
    #            params=dict(name=prefix+"svd-normmaxmin", adjacency_norm="svd", norm_factor=21.46, norm_subtract=2.42),
    #            **jkw)

    ## Test orthogonal matrices
    #submit_slurm_job(
    #        params=dict(name="ortho", distribution="ortho"),
    #        **jkw)

    ## no spectral-dense
    #submit_slurm_job(
    #        params=dict(name="nospectral", input_method="dense"),
    #        **jkw)

    ## add noise...
    #for noise in [1e-8, 1e-6, 1e-4, 1e-2]:

    #    submit_slurm_job(
    #            params=dict(name=f"noise{noise:0.0e}", noise_amplitude=noise),
    #            **jkw)

    #    submit_slurm_job(
    #            params=dict(name=f"tspin-noise{noise:0.0e}", spinup_steps_train=500, noise_amplitude=noise),
    #            **jkw)

    #    submit_slurm_job(
    #            params=dict(name=f"svd-noise{noise:0.0e}", adjacency_norm="svd", noise_amplitude=noise),
    #            **jkw)

    #    submit_slurm_job(
    #            params=dict(name=f"svd-tspin-noise{noise:0.0e}", spinup_steps_train=500, adjacency_norm="svd", noise_amplitude=noise),
    #            **jkw)


    ## A random test...
    #submit_slurm_job(
    #        params=dict(name="tspin-normstd", spinup_steps_train=500, norm_factor=3.67, norm_subtract=2.42),
    #        **jkw)

    ## Another one...
    #submit_slurm_job(
    #        params=dict(name="svd-normstd-quadratic", adjacency_norm="svd", norm_factor=3.67, norm_subtract=2.42, readout_method="quadratic"),
    #        **jkw)

    ## Add timesteps back to tests with tspin discarded
    #submit_slurm_job(
    #        params=dict(name="tspin-spinback", spinup_steps_train=500, time_dimension_train=42_500),
    #        **jkw)


    ## Test what happens when we train on perfect data, predict on data with noise
    #submit_slurm_job(
    #        params=dict(name="testnoise1e-02", add_noise_test=True, noise_amplitude=0.01),
    #        **jkw)

    ## now, discard
    #submit_slurm_job(
    #        params=dict(name="tspin-testnoise1e-02", add_noise_test=True, noise_amplitude=0.01, spinup_steps_train=500),
    #        **jkw)

    ## Test how it goes computing the temporal average misfit, rather than sum
    #submit_slurm_job(
    #        params=dict(name="tavgmisfit", misfit_type="avg"),
    #        **jkw)

    ## Test how it goes computing the temporal average misfit, rather than sum
    #submit_slurm_job(
    #        params=dict(name="svd-normstd-tavgmisfit", adjacency_norm="svd", norm_factor=3.67, norm_subtract=2.42, misfit_type="avg"),
    #        **jkw)

    ## test normalization by 1/sqrt(n_input)
    #submit_slurm_job(
    #        params=dict(name="sqin", input_method="dense-sqin"),
    #        **jkw)

    #submit_slurm_job(
    #        params=dict(name="sqin-normstd", input_method="dense-sqin", norm_factor=3.67, norm_subtract=2.42),
    #        **jkw)

    #submit_slurm_job(
    #        params=dict(name="sqin-svd", input_method="dense-sqin", adjacency_norm="svd"),
    #        **jkw)

    #submit_slurm_job(
    #        params=dict(name="sqin-svd-normstd", input_method="dense-sqin", adjacency_norm="svd", norm_factor=3.67, norm_subtract=2.42),
    #        **jkw)
