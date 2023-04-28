"""transient_startindex is fixed so that...
fix_factor = 4 =>
    n_sub = 1 covers 15/4 years
    n_sub = 4 covers 15 years
    n_sub = 16 covers 60 years

fix_factor = 16 =>
    n_sub = 1 covers 1/16 years
    n_sub = 4 covers 15/4 years
    n_sub = 16 covers 15 year
"""

import os
import numpy as np

import sys
sys.path.append("../..")
from pw import submit_slurm_job

from run_conservation import RCTester


class FixedStepsRCTester(RCTester):

    fix_factor              = None
    use_original_dataset    = None

    @property
    def n_sub(self):
        return self.delta_t // 300

    @property
    def macro_calibration_directory(self):
        out_dir = super().macro_calibration_directory
        out_dir = out_dir.replace("macro-calibration", f"fixed-steps-{self.fix_factor:02d}-macro-calibration")
        if not self.use_original_dataset:
            out_dir = out_dir.replace("fixed-steps", "fixed-steps-long")
        return out_dir


    def years2iters(self, years):
        """In terms of the original model time step!"""
        return int(years * 12 * 30 * 24 * 3600 / 300)


    def days2iters(self, days):
        """In terms of the original model time step!"""
        return int(days * 24 * 3600 / 300)


    def get_new_params(self):
        new_params = super().get_new_params()

        # This is the main difference, because now it chooses a fixed number of timesteps
        # Note that these indices are in terms of the possibly subsampled time dimensio"""n
        new_params["lazydata"]["transient_startindex"]  = int( self.years2iters(  15 ) / self.fix_factor )

        if not self.use_original_dataset:
            new_params["lazydata"]["zstore_path"]           = "/contrib/Tim.Smith/qgrc-teachers/sqg/resolution/long-trajectory/sqg.theta.0300dt.064n.100kt.02z.02y.02x.zarr"

            new_params["lazydata"]["spinup_startindex"]     = self.years2iters(  80 ) // self.n_sub
            new_params["lazydata"]["predict_startindex"]    = (self.years2iters(  80 ) + self.days2iters( 10 )) // self.n_sub
            new_params["lazydata"]["final_index"]           = self.years2iters( 100 ) // self.n_sub
        return new_params


def main(delta_t, cost_terms, fix_factor, use_original_dataset=True, RCTools="MapRCTools", n_samples=50, n_macro=10, n_overlap=1, n_reservoir=6_000):

    for mode in ["macro-calibration", "micro-calibration", "validation"]:
        n_s = None if "validation" not in mode else n_samples
        rct = FixedStepsRCTester(
                delta_t=delta_t,
                cost_terms=cost_terms,
                fix_factor=fix_factor,
                use_original_dataset=use_original_dataset,
                RCTools=RCTools,
                mode=mode,
                n_samples=n_s,
                n_macro=n_macro,
                n_overlap=n_overlap,
                n_reservoir=n_reservoir,
                )
        rct()


if __name__ == "__main__":

    jid = None
    for delta_t in [300, 1200]:
        params = {
                "delta_t"               : delta_t,
                "cost_terms"            : {"nrmse": 1, "spectral": 0.1},
                "fix_factor"            : 16,
                "use_original_dataset"  : True,
                }

        jid = submit_slurm_job("run_fixed_steps", "main", params,
                partition="compute",
                dependency=jid)
