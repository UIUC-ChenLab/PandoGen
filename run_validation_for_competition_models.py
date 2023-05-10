# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from potential_predictions import main as prediction_main
from argparse import Namespace
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from competition_predictions import CompetitionPredictionArgs, read_data
from typing import Optional, Tuple
import os
import json
import sklearn.metrics
import logging

logger = logging.getLogger(__file__)


@dataclass
class ValidationArgs:
    output_prefix: str = field(
        metadata={"help": "Prefix of the output file"}
    )

    count_cutoff: Optional[int] = field(
        default=50,
        metadata={"help": "Count above which we consider the sequence a positive label"}
    )


def main(args: Tuple[CompetitionPredictionArgs, ValidationArgs]):
    comp_args, val_args = args
    validation_file = comp_args.target_sequences
    comp_args.target_sequences = read_data(comp_args.target_sequences)[0]
    results = dict(prediction_main(comp_args))

    with open(validation_file, "r") as fhandle:
        src = [json.loads(x) for x in fhandle]

    y_true = []
    y_predict = []

    for item in src:
        seq = item["seq"]
        count = item["count"]
        potential = results.get(seq, -100)
        y_true.append(count >= val_args.count_cutoff)
        y_predict.append(potential)

    auc_score = sklearn.metrics.roc_auc_score(y_true, y_predict)
    
    with open(val_args.output_prefix, "w") as fhandle:
        fhandle.write(f"AUC (cutoff={val_args.count_cutoff}, checkpoint={comp_args.checkpoint_path}) = {auc_score}")

    logger.info(f"AUC (cutoff={val_args.count_cutoff}, checkpoint={comp_args.checkpoint_path}) = {auc_score}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = HfArgumentParser((CompetitionPredictionArgs, ValidationArgs))
    main(parser.parse_args_into_dataclasses())
