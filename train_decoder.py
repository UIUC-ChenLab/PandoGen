# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    BertLMHeadModel,
    __version__,
)
from data_processing import SimpleSequenceDataset, collate_function_for_decoder
from typing import Tuple, Optional, List
import models
import logging
import os
import pickle
import torch
from collections import defaultdict
from argparse import Namespace
import re
import random
from train import ModelArguments, DataArguments, filter_size_mismatches
import pysam
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(name=__file__)


def make_data(data_args: DataArguments, max_length: int = 1400) -> tuple:
    logger.info("Loading dataset")

    train = SimpleSequenceDataset(data_args.train, max_length=max_length, ignore_too_long=False)
    val = SimpleSequenceDataset(data_args.val, max_length=max_length, ignore_too_long=True)

    return train, val


def make_model(args: ModelArguments) -> BertLMHeadModel:
    logger.info("Initializing model architecture")

    model = models.create_decoder_model(
        tokenizer=None,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act=args.hidden_act,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        num_quark_quantiles=args.num_quark_quantiles,
    )

    if args.checkpoint_params:
        logger.info("Loading pre-trained parameters")
        pretrained = models.from_pretrained(
            args.checkpoint_params,
            args.checkpoint_model_type,
        )
        if args.checkpoint_model_type == "EncoderDecoder":
            logger.info("Loading from EncoderDecoder model")
            tgt_state_dict = model.bert.state_dict()
            state_dict = pretrained.encoder.state_dict()
            filter_size_mismatches(None, state_dict, tgt_state_dict)
            res = model.bert.load_state_dict(state_dict, strict=False)
            logger.warning(f"Following keys weren't loaded {res}")
        else:
            logger.info("Loading from Decoder model")
            tgt_state_dict = model.state_dict()
            state_dict = pretrained.state_dict()
            filter_size_mismatches(None, state_dict, tgt_state_dict)
            res = model.load_state_dict(state_dict, strict=False)
            logger.warning(f"Following keys weren't loaded {res}")

    return model    


def make_trainer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    data_args: DataArguments
) -> Trainer:
    if 0 <= model_args.lr_decay_rate < 1:
        raise NotImplementedError("LR decay rate is not implemented for Decoder")

    model = make_model(model_args)
    data = make_data(data_args, max_length=model_args.max_position_embeddings)

    def subset_helper(data, n):
        return torch.utils.data.Subset(
            data,
            indices=random.sample(range(len(data)), k=n)
        )

    if data_args.max_train_samples:
        train_data = subset_helper(data[0], data_args.max_train_samples)
    else:
        train_data = data[0]

    if data_args.max_val_samples:
        val_data = subset_helper(data[1], data_args.max_val_samples)
    else:
        val_data = data[1]

    return Trainer(
        model,
        args=training_args,
        data_collator=collate_function_for_decoder,
        train_dataset=train_data,
        eval_dataset=val_data,
    ), len(data[0]), len(data[1])


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    random.seed(training_args.seed)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Checkpoint resumption logic from run_clm.py sample script
    # https://github.com/huggingface/transformers/blob/v4.16.2-release/examples/pytorch/language-modeling/run_clm.py
    last_checkpoint = None

    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    trainer, n_train, n_val = make_trainer(model_args, training_args, data_args)
    num_params = sum(x.numel() for x in trainer.model.parameters())
    logger.info(f"Model created with {num_params} parameters")

    if training_args.do_train:
        ### Checkpoint resumption logic from run_clm.py
        # https://github.com/huggingface/transformers/blob/v4.16.2-release/examples/pytorch/language-modeling/run_clm.py
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = n_train
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["val_samples"] = n_val
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s")
    main()
