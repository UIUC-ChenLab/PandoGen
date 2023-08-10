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

    # See note below
    if len(comp_args.target_sequences) % comp_args.embedding_batch_size == 1:
        logger.info("Last batch is singleton, changing batch size")
        comp_args.embedding_batch_size -= 1
        if comp_args.embedding_batch_size < 2:
            raise ValueError("Batch size is less than 2. This won't work.")

    if len(comp_args.target_sequences) % comp_args.predict_batch_size == 1:
        logger.info("Last batch is singleton, changing batch size")
        comp_args.predict_batch_size -= 1
        if comp_args.predict_batch_size < 2:
            raise ValueError("Batch size is less than 2. This won't work.")

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
    """
    Note: If the last batch has size = 1, there will be a failure due to a bug. The following is the
    explanation.

    Since competition_models.Scorer is used, which produces a tensor and not a dict output,
    Trainer treats the output as the variable "logits" (https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L2604)
    Then, for single batch-size cases, Trainer unrolls logits, that is, if logits are of shape [1, 1], it makes it
    into shape [1] (https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L2613)
    Then, the concat operation fails because one of the variables is 2D and another is 1D (i.e. logits): https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L2394

    Due to this, we will change batch_size to be 1 less than provided in main
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = HfArgumentParser((CompetitionPredictionArgs, ValidationArgs))
    main(parser.parse_args_into_dataclasses())
