# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from competition_predictions import (
    CompetitionPredictionArgs,
    SingleDataset,
    read_data,
    simple_collator,
)
from transformers import HfArgumentParser, TrainingArguments, Trainer
import json
import competition_models
import csv
from transformers.file_utils import WEIGHTS_NAME
import torch
import os
import random
import numpy as np


def make_model(args: CompetitionPredictionArgs) -> competition_models.Scorer:
    competition_model = competition_models.make_model(
        args.pretrained_path,
        args.referee_type,
        prediction_type="weight",
        pooler_type=args.pooler_type,
        model_type=args.model_type,
        n_leading_weeks=None,
    )
    weights = torch.load(
        os.path.join(args.checkpoint_path, WEIGHTS_NAME),
        map_location="cpu"
    )
    competition_model.load_state_dict(weights)
    scorer = competition_models.Scorer(competition_model)
    scorer.eval()
    if torch.cuda.is_available():
        scorer.cuda()
    return scorer


def main(args: CompetitionPredictionArgs):
    ## Preliminary: Make model and read data
    scorer = make_model(args)

    if isinstance(args.target_sequences, str):
        sequences, _ = read_data(args.target_sequences, n_leading_weeks=None)
    else:
        sequences = args.target_sequences

    ## Create dataset
    # Sanity-check to make sure there is no unfounded bias due to
    # a specific order in the input
    randindices = list(range(len(sequences)))
    random.shuffle(randindices)
    sequences = [sequences[i] for i in randindices]

    if args.sars_cov2_ref:
        with open(args.sars_cov2_ref, "r") as fhandle:
            ref = fhandle.read().strip()
    else:
        ref = None

    dset = SingleDataset(sequences, ref)

    ## Create prediction loop
    prediction_loop = Trainer(
        model=scorer,
        args=TrainingArguments(
            output_dir="/tmp",
            per_device_eval_batch_size=args.predict_batch_size,
        ),
        data_collator=simple_collator,
    )

    ## Collect test results by running prediction loop
    results = prediction_loop.predict(test_dataset=dset)
    predictions = np.squeeze(results.predictions, axis=1)
    result_tuple = zip(
        sequences, predictions.tolist(),
    )

    ## Write results
    if args.output_path is not None:
        with open(args.output_path, "w", newline="") as fhandle:
            writer = csv.DictWriter(fhandle, fieldnames=["seq0", "potential"])
            writer.writeheader()
            for seq, p in result_tuple:
                writer.writerow({"seq0": seq, "potential": p})

    return result_tuple


if __name__ == "__main__":
    parser = HfArgumentParser((CompetitionPredictionArgs, ))
    args, = parser.parse_args_into_dataclasses()
    main(args)
