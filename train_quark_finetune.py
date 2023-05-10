# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import quark_finetune
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    BertLMHeadModel,
)
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
import pysam
from data_processing import Tokenizer
import torch
import math
import competition_models
import models
import copy
import data_processing
from utils import _DEFAULT_SPECIAL_TOKENS, SpecialTokens
import random
import transformers
import logging
import json
import os
import numpy as np
from transformers.trainer_callback import EarlyStoppingCallback

logger = logging.getLogger(__file__)


@dataclass
class ModelArguments:
    generative_model: str = field(
        metadata={"help": "Generative model that should be fine-tuned"}
    )

    gen_num_return_sequences: int = field(
        metadata={"help": "Batch size of generation"}
    )

    n_init_batches: int = field(
        metadata={"help": "Number of batches to run initialization"}
    )

    n_eval_batches: int = field(
        metadata={"help": "Number of batches to init quantiles and data pool"}
    )

    pool_size: int = field(
        metadata={"help": "Size of data pool for each training epoch"}
    )

    quark_beta: float = field(
        metadata={"help": "KL-divergence co-efficient for quark"}
    )

    potential_model: Optional[str] = field(
        default=None,
        metadata={"help": "Model that measures potentials. If not provided, likelihood model will be used."}
    )

    potential_pretrained_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path of pretrained potential model containing architecture details. If not provided, likelihood model will be used."}
    )

    quark_alpha: float = field(
        default=1,
        metadata={"help": "Alpha co-efficient for quark"}
    )

    quark_scale: float = field(
        default=1,
        metadata={"help": "Scaling for loss"}
    )

    prior_sequences: Optional[str] = field(
        default=None,
        metadata={"help": "List of sequences already reported"}
    )

    quantile_spec: Optional[str] = field(
        default=None,
        metadata={"help": "Quantile specification file"},
    )

    gen_do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Perform sampling instead of greedy"}
    )

    gen_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams for beam search"}
    )

    gen_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Generation temperature"}
    )

    gen_top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k for generation"}
    )

    gen_top_p: Optional[float] = field(
        default=None,
        metadata={"help": "Top-p for nucleus sampling"}
    )

    gen_max_new_tokens: Optional[int] = field(
        default=1400,
        metadata={"help": "Maximum number of new tokens per generation"}
    )

    gen_num_beam_groups: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beam groups for generation"}
    )

    gen_diversity_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "Penalizes lack of diversity among beam groups (only for group beam search)"}
    )

    init_sequences: Optional[str] = field(
        default=None,
        metadata={"help": "Initialization sequences, when provided avoid the need for init loop"}
    )

    test_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Held-out dataset to calculate likelihood on"}
    )

    use_stratified_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Use stratified sampling in quark training"}
    )

    no_dropout: Optional[bool] = field(
        default=False,
        metadata={"help": "Set all dropout to 0"}
    )

    early_stopping: Optional[int] = field(
        default=None,
        metadata={"help": "Number of early stopping iterations if desired"}
    )


def set_dropout_0(model: torch.nn.Module):
    """
    Set dropout to 0 for all layers
    """
    for m in model.children():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0
        else:
            set_dropout_0(m)


def make_model(
    args: ModelArguments,
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS
) -> quark_finetune.QuarkModel:

    decoder = models.from_pretrained(args.generative_model, model_type="Decoder")
    ref_model = models.from_pretrained(args.generative_model, model_type="Decoder")

    if args.no_dropout:
        logger.info("Setting dropout to 0 for training model")
        set_dropout_0(decoder)

    if args.potential_model and args.potential_pretrained_path:
        competition_model = competition_models.make_model(
            pretrained_path=args.potential_pretrained_path,
            referee_type="binary",
            prediction_type="weight",
            pooler_type=None,
            model_type="Decoder",
            n_leading_weeks=None,
            mix_leading_weeks=None,
        )
        weights = torch.load(
            os.path.join(args.potential_model, "pytorch_model.bin"),
            map_location="cpu"
        )
        competition_model.load_state_dict(weights)
        scorer = competition_models.Scorer(competition_model)
    else:
        scorer = quark_finetune.LikelihoodScorer(decoder)

    prior_sequences = quark_finetune.tensorize_prior_sequences(
        args.prior_sequences, max_length=args.gen_max_new_tokens + 1)

    mapper = data_processing.Tokenizer().mapper
    eos_token = mapper[special_tokens.end_of_sequence]

    reward_model = quark_finetune.RewardModel(
        scorer=scorer, eos_token=eos_token, prev_sequences=prior_sequences)

    quark_model = quark_finetune.QuarkModel(
        train_model=decoder,
        ref_model=ref_model,
        reward_model=reward_model,
        bos_token_ref=mapper[special_tokens.start_of_sequence],
    )

    return quark_model


def get_gen_kwargs(args: ModelArguments) -> dict:
    return {
        key[4:]: value for key, value in asdict(args).items() if key[:4] == "gen_"
    }


def make_trainer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
):
    # Create model
    quark_model = make_model(model_args)
    gen_kwargs = get_gen_kwargs(model_args)

    # Create models and quantiles
    with open(model_args.quantile_spec, "r") as fhandle:
        quantile_spec = json.load(fhandle)

    data_pool, quantiles = quark_finetune.init_training(
        quark_model,
        quantile_spec,
        n_batches_init=model_args.n_init_batches,
        gen_kwargs=gen_kwargs,
        pool_size=model_args.pool_size,
        max_length=model_args.gen_max_new_tokens + 1,
        args=training_args,
        pregen=model_args.init_sequences,
        use_stratified_sample=model_args.use_stratified_sample,
    )

    logger.info(f"Obtained quantiles = {quantiles}")

    # Add quantities to training_args
    mapper = Tokenizer().mapper
    quantile_offset = len(mapper)
    training_args.gen_kwargs = gen_kwargs
    training_args.eos_token = mapper[special_tokens.end_of_sequence]
    training_args.quantile_offset = quantile_offset
    training_args.quantiles = quantiles
    training_args.n_eval_steps = model_args.n_eval_batches
    training_args.quark_alpha = model_args.quark_alpha
    training_args.quark_beta = model_args.quark_beta
    training_args.quark_scale = model_args.quark_scale
    training_args.model_args = model_args

    # Correct items for early stopping
    if model_args.early_stopping is not None:
        training_args.load_best_model_at_end = True
        training_args.greater_is_better = True
        training_args.metric_for_best_model = "eval_loss"
        callbacks = [EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping)]
        logger.info("Initializing EarlyStoppingCallback")
    else:
        callbacks = None

    # Create QuarkTrainer
    trainer = quark_finetune.QuarkTrainer(
        model=quark_model,
        args=training_args,
        data_collator=quark_finetune.collate_function,
        train_dataset=data_pool,
        eval_dataset=data_pool,
        callbacks=callbacks,
    )

    return trainer


def compute_ll(
    model: torch.nn.Module, inputs: dict, offset: int,
):
    logits = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    ).logits[:, offset - 1: -1]
    seq = inputs["input_ids"][:, offset:]
    mask = inputs["attention_mask"][:, offset:]

    dist = torch.distributions.categorical.Categorical(logits=logits)
    ll_per_token = dist.log_prob(seq).masked_fill(mask == 0, 0)
    ll_per_seq = torch.sum(ll_per_token, dim=1)
    n_tokens_per_seq = torch.sum(mask, dim=1).float()

    loss = -torch.sum(ll_per_seq) / torch.sum(n_tokens_per_seq)

    return loss, ll_per_seq, n_tokens_per_seq


class QuarkTestLoop(Trainer):
    """
    ad-hoc quark test loop
    """
    def compute_loss(
        self, model: quark_finetune.QuarkModel, inputs: dict, return_outputs: bool = False):

        loss_ref, ll_per_seq_ref, n_tokens_per_seq_ref = compute_ll(
            model=model.ref_model,
            inputs=inputs["ref_sequences"],
            offset=1,
        )

        loss_train, ll_per_seq_train, n_tokens_per_seq_train = compute_ll(
            model=model.train_model,
            inputs=inputs["train_sequences"],
            offset=2,
        )

        rewards = model.get_rewards(
            input_ids=inputs["ref_sequences"]["input_ids"],
            attention_mask=inputs["ref_sequences"]["attention_mask"],
        )[0]

        outputs = torch.stack(
            (
                ll_per_seq_train,
                ll_per_seq_ref,
                rewards[:, 0],
                n_tokens_per_seq_ref,
            ),
            dim=1
        )

        assert(torch.all(n_tokens_per_seq_ref == n_tokens_per_seq_train))

        if return_outputs:
            return loss_train, outputs
        else:
            return loss


class TestLoopDataset(torch.utils.data.Dataset):
    def __init__(self, test_fa: str, bos_token: int, quantile_token: int):
        super().__init__()

        self.sequences = []
        self.mapper = data_processing.Tokenizer().mapper
        self.bos_token = bos_token
        self.quantile_token = quantile_token

        with pysam.FastaFile(test_fa) as fhandle:
            for r in fhandle.references:
                s = fhandle.fetch(r)
                if "stop" in s:
                    continue
                self.sequences.append(s)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        ref_tokenized = [self.bos_token] + [self.mapper[i] for i in seq]
        train_tokenized = [self.bos_token, self.quantile_token] + [self.mapper[i] for i in seq]
        return torch.LongTensor(ref_tokenized), torch.LongTensor(train_tokenized)


def collate_function_test_loop(batch: list):
    max_length = max(x.shape[0] for x in batch)
    ids = torch.zeros(len(batch), max_length).long()
    attn = torch.zeros(len(batch), max_length).byte()
    for i, b in enumerate(batch):
        ids[i, :b.shape[0]] = b
        attn[i, :b.shape[0]] = 1
    return {"input_ids": ids, "attention_mask": attn}


def collate_function_top(batch: list):
    ref_sequences, train_sequences = tuple(zip(*batch))
    return {
        "ref_sequences": collate_function_test_loop(ref_sequences),
        "train_sequences": collate_function_test_loop(train_sequences),
        "labels": torch.Tensor([0]),
    }


def main(model_args: ModelArguments, training_args: TrainingArguments):
    random.seed(training_args.seed)
    log_level = training_args.get_process_log_level()
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s:%(message)s")
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    trainer = make_trainer(model_args, training_args)

    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(trainer.train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["val_samples"] = model_args.n_eval_batches
        metrics = {key: str(value) if type(value) is list else value for key, value in metrics.items()}
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        mapper = data_processing.Tokenizer().mapper
        bos_token = mapper[_DEFAULT_SPECIAL_TOKENS.start_of_sequence]
        quantile_token = len(mapper) + len(training_args.quantiles) - 1
        test_data = TestLoopDataset(
            model_args.test_dataset,
            bos_token=bos_token,
            quantile_token=quantile_token,
        )
        prediction_loop = QuarkTestLoop(
            model=trainer.model,
            args=trainer.args,
            eval_dataset=test_data,
            data_collator=collate_function_top,
        )
        outputs = prediction_loop.predict(test_dataset=test_data).predictions

        ll_per_seq_train = np.exp(outputs[:, 0])
        ll_per_seq_ref = np.exp(outputs[:, 1])
        rewards = outputs[:, 2]
        n_tokens = np.add.reduce(outputs[:, 3])

        weighted_rewards_train = np.add.reduce(ll_per_seq_train * rewards) / rewards.shape[0]
        weighted_rewards_ref = np.add.reduce(ll_per_seq_ref * rewards) / rewards.shape[0]
        avg_ll_train = np.add.reduce(ll_per_seq_train) / n_tokens
        avg_ll_ref = np.add.reduce(ll_per_seq_ref) / n_tokens

        logger.info(
            f"Weighted rewards train = {weighted_rewards_train}\n"
            f"Weighted rewards ref   = {weighted_rewards_ref}\n"
            f"Average LL train       = {avg_ll_train}\n"
            f"Average LL ref         = {avg_ll_ref}\n"
        )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, training_args)
