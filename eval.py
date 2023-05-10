# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field, asdict
from data_processing import Dataset, collate_function, Tokenizer, CompoundDataset
import data_processing
from typing import Tuple, Optional, List
import models
import logging
import os
import pickle
from train import make_trainer, make_data, ModelArguments, DataArguments, load_from_checkpoint
import torch

logger = logging.getLogger(name=__file__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info("Initializing data, models and trainer")
    data = make_data(data_args)
    train_dataset, val_dataset, test_dataset = data
    model = models.from_pretrained(model_args.checkpoint_params)
    trainer = make_trainer(model, data, training_args, model_args)
    num_params = sum(x.numel() for x in model.parameters())
    logger.info(f"Loaded model with {num_params} parameters")

    if data_args.max_val_samples:
        val_dataset.set_length(data_args.max_val_samples)

    if model_args.checkpoint_params is None:
        raise ValueError("No checkpoints provided to evaluate model")

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["val_samples"] = len(val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s")
    main()
