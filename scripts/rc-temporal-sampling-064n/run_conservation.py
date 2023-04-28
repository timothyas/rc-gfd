import os
import numpy as np
from dask_jobqueue import SLURMCluster
from ddc import LazyCalValDriver, YAMLParser

import sys
sys.path.append("../..")
from pw import submit_slurm_job

class RCTester():

    delta_t     = None
    cost_terms  = None
    mode        = None
    n_samples   = None
    n_macro     = 10
    RCTools     = "MapRCTools"
    config      = "config-rc.yaml"
    n_workers   = 8
    n_overlap   = 1
    n_reservoir = 6_000

    @property
    def n_nodes(self):
        if "macro" in self.mode:
            return 10
        else:
            return 3 if self.delta_t == 300 else 2


    @property
    def output_directory(self):

        if "macro" in self.mode:
            return self.macro_calibration_directory

        elif "micro" in self.mode:
            return self.micro_calibration_directory

        elif "validation" in self.mode:
            return self.validation_directory

        else:
            raise ValueError


    @property
    def macro_calibration_directory(self):
        cstr = "cost-"
        if "RK4" in self.RCTools:
            cstr += "rk4-"

        if self.n_macro != 10:
            cstr += f"{self.n_macro:03d}macro-"

        if self.n_overlap > 1:
            cstr += f"{self.n_overlap:02d}overlap-"

        if self.n_reservoir != 6_000:
            cstr += f"{self.n_reservoir//1000}kNr-"

        cstr += "-".join(f"{k}{v:1.1e}" for k,v in self.cost_terms.items())

        out_dir = f"{cstr}/macro-calibration-{self.delta_t:04d}dt"
        return out_dir


    @property
    def micro_calibration_directory(self):
        return self.macro_calibration_directory.replace("macro-calibration", "micro-calibration")


    @property
    def validation_directory(self):
        out_dir = self.macro_calibration_directory.replace("macro-calibration", "validation")
        return out_dir + f"-{self.n_samples:03d}samples"


    @property
    def log_directory(self):
        return f"/home/Tim.Smith/sqg/resolution/rc-temporal-sampling-064n/{self.output_directory}"


    @property
    def sample_indices(self):
        if self.n_samples is None:
            return None
        else:
            sdict = self.yp.read("../sample-indices.yaml")
            return sdict[self.n_samples][self.delta_t]

    @property
    def parameters(self):
        return ("sigma", "spectral_radius", "sigma_bias", "leak_rate", "tikhonov_parameter")


    @property
    def is_complete(self):
        if "macro" in self.mode:
            return os.path.isfile(f"{self.output_directory}/config-optim.yaml")
        elif "micro" in self.mode:
            return os.path.isdir(f"{self.output_directory}/rcmodel.zarr")
        elif "validation" in self.mode:
            return os.path.isdir(f"{self.output_directory}/results.{self.n_samples-1:03d}.zarr")
        else:
            return False


    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            try:
                getattr(self, key)
            except:
                raise
            setattr(self, key, val)

        self.yp = YAMLParser()

        if "macro" in self.mode and not self.is_complete:
            sample_file = f"{self.macro_calibration_directory}/config-sample.yaml"
            if os.path.isfile(sample_file):
                self.config = sample_file


    def get_optimized_rc(self):

        optim_file = f"{self.macro_calibration_directory}/config-optim.yaml"

        if os.path.isfile(optim_file):
            co = self.yp.read(optim_file)
            rc_params = {k: co["rc"][k] for k in self.parameters}

        else:
            # otherwise get best case from sampling
            sample_file = optim_file.replace("config-optim", "config-sample")
            cs = self.yp.read(sample_file)
            sample = np.array(cs["calibration"]["initial_sample"])
            cost = cs["calibration"]["initial_cost"]
            index = np.argmin(cost)
            rc_params = {}
            for key, val in zip(cs["calibration"]["parameters"], sample[:,index]):

                is_log10 = key[:6] == "log10_"
                pname = key if not is_log10 else key[6:]
                v = val if not is_log10 else 10.**val
                rc_params[pname] = float(v)

        return rc_params


    def get_new_params(self):

        new_params = {
                "data" : {
                    "delta_t"                   : self.delta_t,
                    },
                "lazydata" : {
                    "delta_t"                   : self.delta_t,
                    "train_startindex"          : 0,
                    "transient_startindex"      : int(1_555_200 / self.delta_t * 300),
                    "spinup_startindex"         : int(2_073_600 / self.delta_t * 300),
                    "predict_startindex"        : int(2_076_480 / self.delta_t * 300),
                    "final_index"               : int(2_592_000 / self.delta_t * 300),
                    "local_halo"                : self.n_overlap,
                    },
                "rc" : {
                    "RCTools"                   : self.RCTools,
                    "spinup_steps_predict"      : int(2_880 / self.delta_t * 300),
                    "spinup_steps_train"        : 0,
                    "reservoir_dimension"       : self.n_reservoir,
                    },
                "compute" : {
                    "n_nodes"                   : self.n_nodes,
                    },
                "calibration" : {
                    "cost_terms"                : self.cost_terms,
                    "forecast_steps"            : int( 144 / self.delta_t * 300 ) + 1,
                    "n_macro"                   : self.n_macro,
                    },
                "validation" : {
                    "forecast_steps"            : int( 144 / self.delta_t * 300 ) + 1,
                    "sample_indices"            : self.sample_indices,
                    },
                }

        if self.delta_t != 300:
            dt_step = self.delta_t // 300
            new_params["lazydata"]["temporal_subsampling"] = {"start": None, "stop": None, "step": dt_step}

        if "micro" in self.mode:
            new_params["rc"]["store_to_zarr"] = True
            new_params["rc"]["read_from_zarr"] = False
            new_params["rc"].update(self.get_optimized_rc())

        if "validation" in self.mode:
            new_params["rc"]["store_to_zarr"] = False
            new_params["rc"]["read_from_zarr"] = True
            new_params["rc"]["zstore_input_path"] = f"{self.micro_calibration_directory}/rcmodel.zarr"
            new_params["rc"].update(self.get_optimized_rc())

        return new_params


    def __call__(self):

        if not self.is_complete:

            driver = LazyCalValDriver(config=self.config, output_directory=self.output_directory)

            driver.overwrite_params(self.get_new_params())
            client = driver.create_client(
                    Cluster=SLURMCluster,
                    cluster_kwargs={
                        "log_directory" : self.log_directory,
                        "processes"     : self.n_workers
                        },
                    )

            if "macro" in self.mode:
                driver.run_calibration(client=client)

            if "micro" in self.mode:
                driver.run_micro_calibration(client=client)

            if "validation" in self.mode:
                driver.run_validation(client=client)

            client.close()


def main(delta_t, cost_terms, RCTools="MapRCTools", n_samples=50, n_macro=10, n_overlap=1, n_reservoir=6_000):

    for mode in ["macro-calibration", "micro-calibration", "validation"]:
        n_s = None if "validation" not in mode else n_samples
        rct = RCTester(
                delta_t=delta_t,
                cost_terms=cost_terms,
                RCTools=RCTools,
                mode=mode,
                n_samples=n_s,
                n_macro=n_macro,
                n_overlap=n_overlap,
                n_reservoir=n_reservoir
                )
        rct()


if __name__ == "__main__":

    jid = None
    for delta_t in [14400, 4800, 1200, 300]:
        for cost_terms in [
                {"nrmse": 1},
                # KE RMSE
                {"nrmse": 1, "totspectral": 1.e-5},
                {"nrmse": 1, "totspectral": 1.e-4},
                {"nrmse": 1, "totspectral": 1.e-3},
                # KE NRMSE
                {"nrmse": 1, "spectral": 0.001},
                {"nrmse": 1, "spectral": 0.01},
                {"nrmse": 1, "spectral": 0.1},
                {"nrmse": 1, "spectral": 1.0},
                {"nrmse": 1, "spectral": 10.0},
                # Global Integral is not a good idea
                #{"nrmse": 1, "global-integral": 0.0001},
                ]:
            params = {
                    'delta_t'       : delta_t,
                    'cost_terms'    : cost_terms,
                    }

            jid = submit_slurm_job("run_conservation", "main", params,
                                   partition="spot",
                                   dependency=jid)
