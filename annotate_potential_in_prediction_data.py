# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
# from competition_predictions import main as prediction_main
from potential_predictions import main as prediction_main
from competition_predictions import CompetitionPredictionArgs
import json
from typing import Union
import os
import shutil
from copy import deepcopy
import csv
from transformers import HfArgumentParser
import logging
from dataclasses import asdict


def collect_sequences(filename: str) -> set:
    sequences_to_evaluate = set()

    with open(filename, "r") as fhandle:
        for line in fhandle:
            parsed = json.loads(line)

            if "[" in parsed["seq"] or "]" in parsed["seq"]:
                continue

            sequences_to_evaluate.add(parsed["seq"])

    return sequences_to_evaluate


def create_dset_for_prediction(sequences: Union[set, list], workdir: str) -> None:
    filename = None

    with open(os.path.join(workdir, "potential_calc_inputs.json"), "w") as fhandle:
        filename = fhandle.name

        for item in sequences:
            fhandle.write(json.dumps({"seq": item, "weekly_counts": [0]}) + "\n")

    return filename


def collect_results(prediction_file: str) -> dict:
    results = dict()

    with open(prediction_file, "r", newline="") as fhandle:
        reader = csv.DictReader(fhandle)
        for row in reader:
            seq = row["seq0"]
            potential = row["potential"]
            results[seq] = potential

    return results


def annotate_results(src_file: str, results: dict, output_file: str) -> None:
    with open(src_file, "r") as fhandle, open(output_file, "w") as whandle:
        for line in fhandle:
            items = json.loads(line)
            if items["seq"] in results:
                items["metric"] = results[items["seq"]]
                whandle.write(json.dumps(items) + "\n")


def get_basefilename(f: str) -> str:
    return os.path.split(os.path.splitext(f)[0])[1]


def main(args: CompetitionPredictionArgs):
    # 0. Prepare workdir
    args.workdir = os.path.join(
        args.workdir, f"temp_{get_basefilename(args.target_sequences)}"
    )

    if os.path.exists(args.workdir):
        shutil.rmtree(args.workdir)

    os.makedirs(args.workdir)

    # 1. Collect all sequences to be be predicted for
    sequences = collect_sequences(args.target_sequences)

    # 2. Prepare prediction data
    prediction_data = create_dset_for_prediction(sequences, workdir=args.workdir)

    # 3. Perform prediction
    new_args = deepcopy(args)
    new_args.output_path = os.path.join(args.workdir, "potential_predictions.csv")
    new_args.target_sequences = prediction_data
    prediction_main(new_args)    

    # 4. Collect results
    results = collect_results(new_args.output_path)

    # 5. Annotate results
    annotate_results(args.target_sequences, results, args.output_path)

    # 6. Cleanup
    shutil.rmtree(args.workdir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = HfArgumentParser((CompetitionPredictionArgs, ))
    args, = parser.parse_args_into_dataclasses()
    main(args)
