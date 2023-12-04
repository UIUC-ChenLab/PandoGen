# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import torch
import pysam
from transformers import TrainingArguments, HfArgumentParser, Trainer
from dataclasses import dataclass, field
import logging
import random
import transformers
import os
from transformers.trainer_utils import get_last_checkpoint
import tqdm
import sys
from tokenizers import Tokenizer
import re
from typing import Optional
from transformers.trainer_callback import EarlyStoppingCallback

logger = logging.getLogger(__file__)


@dataclass
class ModelArguments:
    pretrained_model_path: str = field(
        metadata={"help": "Path of pretrained model"}
    )

    training_file: str = field(
        metadata={"help": "Training fasta"}
    )

    validation_file: str = field(
        metadata={"help": "Validation fasta"}
    )

    progen_install: str = field(
        metadata={"help": "Path to progen2 source code"}
    )

    early_stopping: Optional[int] = field(
        default=None,
        metadata={"help": "Early stopping patience"}
    )

    force_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Overwrite max-length for the progen model"}
    )


def get_model(model_args: ModelArguments):
    logger.info("################### Loading ProGen model ####################")
    sys.path.append(model_args.progen_install)
    import progen2.models.progen.modeling_progen as modeling_progen
    ProGenForCausalLM = modeling_progen.ProGenForCausalLM
    model = ProGenForCausalLM.from_pretrained(model_args.pretrained_model_path)
    is_cuda = next(model.parameters()).is_cuda
    device = next(model.parameters()).get_device()

    if model_args.force_max_length:
        logger.info("###################### Changing model max_positions ######################")
        for m in model.modules():
            if isinstance(m, modeling_progen.ProGenAttention):
                if hasattr(m, "bias"):
                    old_bias_size = m.bias.shape
                    max_positions = model_args.force_max_length
                    # See https://github.com/salesforce/progen/blob/485b2ea3db98f8d65d0cd86c2c85ae639b37a678/progen2/models/progen/modeling_progen.py#L65
                    bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                        1, 1, max_positions, max_positions
                    )
                    if is_cuda:
                        bias = bias.to(device)
                    m.register_buffer("bias", bias)
                    logger.info(f"Changed bias for module from {old_bias_size} to {m.bias.shape}")

    return model


class Dataset(torch.utils.data.Dataset):
    def __init__(self, fasta: str, progen_src: str):
        super().__init__()

        with pysam.FastaFile(fasta) as fhandle:
            self.references = [
                r for r in tqdm.tqdm(
                    fhandle.references,
                    desc="Reading files") if "stop" not in fhandle.fetch(r)]

        with open(os.path.join(progen_src, "progen2", "tokenizer.json"), "r") as fhandle:
            self.tokenizer = Tokenizer.from_str(fhandle.read())

        self.fasta = fasta
        self._fhandle = None

    @property
    def fhandle(self):
        if self._fhandle is None:
            self._fhandle = pysam.FastaFile(self.fasta)

        return self._fhandle

    def __len__(self):
        return len(self.references)

    def __getitem__(self, idx: int) -> tuple:
        seq = self.fhandle.fetch(self.references[idx])
        tokenized = self.tokenizer.encode(f"1{seq}2").ids
        return tokenized


def collate_function(batch: list) -> dict:
    """
    ProtGPT2 is an instgance of GPT2LMHeadModel, and the labels are auto-shifted
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/gpt2/modeling_gpt2.py#L1072
    """
    max_length = max(len(x) for x in batch)
    input_ids = torch.zeros(len(batch), max_length).long()
    attention_mask = torch.zeros(len(batch), max_length).byte()

    for i, l in enumerate(batch):
        input_ids[i, :len(l)] = torch.LongTensor(l)
        attention_mask[i, :len(l)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def main(training_args: TrainingArguments, model_args: ModelArguments):
    random.seed(training_args.seed)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    model = get_model(model_args)
    train_dataset = Dataset(model_args.training_file, progen_src=model_args.progen_install)
    val_dataset = Dataset(model_args.validation_file, progen_src=model_args.progen_install)
    n_train = len(train_dataset)
    n_val = len(val_dataset)

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

    if model_args.early_stopping:
        # training_args.load_best_model_at_end = True
        # training_args.metric_for_best_model = "eval_loss"
        callbacks = [EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping)]
        logger.info("Initializing early-stopping callback")
    else:
        callbacks = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_function,
        callbacks=callbacks,
    )

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
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments)
    )
    model_args, training_args = parser.parse_args_into_dataclasses()
    main(training_args, model_args)
