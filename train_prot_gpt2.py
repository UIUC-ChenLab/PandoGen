# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from transformers import AutoModelForCausalLM, AutoTokenizer
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


logger = logging.getLogger(__file__)


@dataclass
class ModelArguments:
    training_file: str = field(
        metadata={"help": "Training fasta"}
    )

    validation_file: str = field(
        metadata={"help": "Validation fasta"}
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, fasta: str):
        super().__init__()

        with pysam.FastaFile(fasta) as fhandle:
            self.references = [
                r for r in tqdm.tqdm(
                    fhandle.references,
                    desc="Reading files") if "stop" not in fhandle.fetch(r)]

        self.tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        self.bos_token = self.tokenizer(self.tokenizer.bos_token)["input_ids"][0]

        # self.tokenized_data = [
        #     [bos_token] + tokenizer(seq)["input_ids"] for seq in \
        #         sequences
        # ]
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
        tokenized = [self.bos_token] + self.tokenizer(seq)["input_ids"]
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

    model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
    train_dataset = Dataset(model_args.training_file)
    val_dataset = Dataset(model_args.validation_file)
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_function,
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
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments)
    )
    model_args, training_args = parser.parse_args_into_dataclasses()
    main(training_args, model_args)
