# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import sklearn
from sklearn.model_selection import ParameterGrid
import json
import argparse
import os


def get_param_string(grid_conf: str):
    string = " ".join(
        f"--{key} {value}" if type(value) is not bool else \
            f"--{key}" for key, value in grid_conf.items() if value is not None
    )
    return string


def get_header(job_prefix: str, idx: int, partition: str, time: int) -> str:
    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name=\"{job_prefix}{idx}\"\n"
    header += f"#SBATCH --output=\"{job_prefix}{idx}.%j.%N.out\"\n"
    header += f"#SBATCH --error=\"{job_prefix}{idx}.%j.%N.err\"\n"
    header += f"#SBATCH --partition={partition}\n"
    header += f"#SBATCH --time={time}\n"
    header += f"module load opence/1.6.1\n"
    return header


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare grid search commands")

    parser.add_argument("--cmd", help="Command name", required=True)
    parser.add_argument("--config", help="Grid config", required=True)
    parser.add_argument("--workdir", help="Working directory", required=True)
    parser.add_argument("--job_prefix", help="Job name prefix", required=True)
    parser.add_argument("--partition", help="Partition", default="gpux1")
    parser.add_argument("--time", help="Number of hours", type=int, default=24)
    parser.add_argument("--output_prefix", help="Prefix of outputs", required=True)
    parser.add_argument("--logging_prefix", help="Logging prefix", required=True)

    args = parser.parse_args()

    with open(args.config, "r") as fhandle:
        grid_config = json.load(fhandle)

    for i, grid_param in enumerate(ParameterGrid(grid_config)):
        grid_param["output_dir"] = f"{args.output_prefix}{i}"
        grid_param["logging_dir"] = f"{args.logging_prefix}{i}"
        param_string = get_param_string(grid_param)
        header = get_header(args.job_prefix, i, args.partition, args.time)
        with open(os.path.join(args.workdir, f"cmd{i}.sh"), "w") as fhandle:
            fhandle.write(header)
            fhandle.write(f"{args.cmd} {param_string}")
