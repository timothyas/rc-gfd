"""Submit on PW GCP"""

import os
import subprocess

def parse_for_jobid(output):
    stdout = output.stdout.decode("utf-8").replace("\n","")

    try:
        jobid = int(stdout.split(" ")[-1])
    except ValueError as err:
        jobid = None
        err.args += (f"Could not convert last word of string to int: ", stdout)
        raise err

    return jobid


def submit_slurm_job(module_name, function_name, params, partition="compute", cpus_per_task=30, dependency=None, dependency_type="afterany"):
    """Generic script to submit a slurm job which calls
    "python -c 'from {module_name} import {function_name} ; {function_name}(**params)'"

    Args:
        module_name (str): name of module calling from
        function name (str): name of function within module to run
        params (dict): with options to pass to the function
        partition (str, optional): specify the partition/queue to
            submit to, e.g. "compute" or "spot"
    """

    long_name = "slurm"
    for key, val in params.items():


        if isinstance(val, dict):
            vstr = "_".join(f"{k}-{v}" for k,v in val.items())

        else:
            vstr = str(val)

        long_name += f"_{key}-{vstr}"

    out_name = long_name.replace("slurm_","slurm/")
    submit_name = long_name.replace("slurm", "submit") + ".sh"
    long_name = long_name.replace("slurm_", "")
    contents = f"#!/bin/bash\n\n"+\
        f"#SBATCH -J {long_name}\n" +\
        f"#SBATCH -o {out_name}.%j.out\n" +\
        f"#SBATCH -e {out_name}.%j.err\n" +\
        f"#SBATCH --nodes=1\n" +\
        f"#SBATCH --ntasks=1\n" +\
        f"#SBATCH --cpus-per-task={cpus_per_task}\n" +\
        f"#SBATCH --partition={partition}\n"+\
        f"#SBATCH -t 120:00:00\n\n"

    contents += "source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"
    contents += "conda activate ddc10\n"
    contents += f"python -c 'from {module_name} import {function_name} ; {function_name}("

    for key, val in params.items():
        if isinstance(val, str):
            contents += f'{key}="{val}",'
        elif isinstance(val, dict):
            contents += '%s={' % key
            for k2,v2 in val.items():
                contents += f'"{k2}": {v2},'

            contents = contents.removesuffix(",")
            contents += '},'
        else:
            contents += f'{key}={val},'

    contents = contents.removesuffix(",")
    contents += ")'\n"

    with open(submit_name, "w") as f:
        f.write(contents)

    if not os.path.isdir("slurm"):
        os.makedirs("slurm")

    # --- Work out any dependencies
    dependstr = ""
    if dependency is not None:
        dependstr = "--dependency="
        if isinstance(dependency, int):
            dependency = [dependency]

        for jid in dependency:
            dependstr += f"{dependency_type}:{jid},"
        dependstr = dependstr[:-1]

    pout = subprocess.run(f"sbatch {dependstr} {submit_name}", shell=True, capture_output=True)
    jobid = parse_for_jobid(pout)
    print(f"Submitted SLURM job: {jobid} {dependstr}")

    return jobid
