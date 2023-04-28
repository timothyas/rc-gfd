
from distributed import LocalCluster
from dask_jobqueue import SLURMCluster
from ddc import LazyCalValDriver, YAMLParser

import sys
sys.path.append("../..")
from pw import submit_slurm_job

class NVARTester():

    delta_t     = None
    n_lag       = None
    order       = None
    n_samples   = None
    mode        = None

    # defaults, these don't change...
    n_neighbors = 1
    config      = "config-nvar.yaml"

    @property
    def output_directory(self):
        outer = f"lag{self.n_lag:02d}-order{self.order:02d}-no{self.n_neighbors:02d}/{self.mode}-{self.delta_t:04d}dt"
        if "cal" in self.mode:
            return outer
        else:
            return outer+ f"-{self.n_samples:03d}samples"

    @property
    def sample_indices(self):
        if "cal" in self.mode:
            return None
        else:
            return self.yp.read("../sample-indices.yaml")[self.n_samples][self.delta_t]


    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            try:
                getattr(self, key)
            except:
                raise
            setattr(self, key, val)

        self.yp = YAMLParser()


    def get_cluster(self):

        base_dir = f"/home/Tim.Smith/sqg/resolution/nvar-temporal-sampling-064n/"
        if self.n_lag < 2:

            ckw = {
                    "local_directory": base_dir+self.output_directory,
                    }
            Cluster = LocalCluster
            n_nodes = 1

        else:
            ckw = {
                    "log_directory" : base_dir+self.output_directory,
                    "processes"     : 6
                    }
            Cluster = SLURMCluster

            n_nodes = 4 if self.n_lag < 3 else 8

        return Cluster, n_nodes, ckw


    def get_new_params(self):

        orderlist = list(x for x in range(self.order+1))

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
                    "local_halo"                : self.n_neighbors,
                    },
                "validation" : {
                    "forecast_steps"            : int( 144 / self.delta_t * 300 ) + 1,
                    "n_samples"                 : self.n_samples,
                    "sample_indices"            : self.sample_indices,
                    },
                "rc" : {
                    "n_lag"                     : self.n_lag,
                    "n_neighbors"               : self.n_neighbors,
                    "order"                     : orderlist,
                    },
                "compute" : {},
                }

        if self.delta_t != 300:
            dt_step = self.delta_t // 300
            new_params["lazydata"]["temporal_subsampling"] = {"start": None, "stop": None, "step": dt_step}

        if self.mode == "validation":
            new_params["rc"]["store_to_zarr"] = False
            new_params["rc"]["read_from_zarr"] = True
            mc_output = self.output_directory.replace("validation", "micro-calibration")
            mc_output = mc_output.replace(f"-{self.n_samples:03d}samples", "")
            zstore = f"{mc_output}/rcmodel.zarr"
            new_params["rc"]["zstore_input_path"] = zstore

        else:
            new_params["rc"]["store_to_zarr"] = True
            new_params["rc"]["read_from_zarr"] = False

        return new_params


def val(delta_t, n_lag, order, n_neighbors, n_samples):

    nvt = NVARTester(
            delta_t=delta_t,
            n_lag=n_lag,
            order=order,
            n_neighbors=n_neighbors,
            n_samples=n_samples,
            mode="validation")

    # --- Create driver and launch
    driver = LazyCalValDriver(config=nvt.config, output_directory=nvt.output_directory)
    new_params = nvt.get_new_params()
    Cluster, n_nodes, ckw = nvt.get_cluster()
    new_params["compute"]["n_nodes"] = n_nodes
    driver.overwrite_params(new_params)

    client = driver.create_client(Cluster=Cluster, cluster_kwargs=ckw)

    client.amm.start()
    driver.run_validation(client=client)


def micro_cal(delta_t, n_lag, order, n_neighbors):

    nvt = NVARTester(
            delta_t=delta_t,
            n_lag=n_lag,
            order=order,
            n_neighbors=n_neighbors,
            n_samples=1,
            mode="micro-calibration")

    # --- Create driver and launch
    driver = LazyCalValDriver(config=nvt.config, output_directory=nvt.output_directory)
    new_params = nvt.get_new_params()
    Cluster, n_nodes, ckw = nvt.get_cluster()
    new_params["compute"]["n_nodes"] = n_nodes
    driver.overwrite_params(new_params)

    client = driver.create_client(Cluster=Cluster, cluster_kwargs=ckw)

    client.amm.start()
    driver.run_micro_calibration(client=client)


if __name__ == "__main__":

    jid = None
    n_samples = 50
    for n_lag in [0, 1, 2, 3]:
        for delta_t in [300, 1200, 4800]:
            params = {
                    'delta_t'       : delta_t,
                    'n_lag'         : n_lag,
                    'order'         : 2,
                    'n_neighbors'   : 1,
                    }

            jid = submit_slurm_job("run_validation", "micro_cal", params,
                                   partition="spot",
                                   dependency=jid,
                                   dependency_type="afterany")

            params["n_samples"] = n_samples
            jid = submit_slurm_job("run_validation", "val", params,
                                   partition="spot",
                                   dependency=jid,
                                   dependency_type="afterok")
