# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import torch
import sequence_feature_data_utils
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import HfArgumentParser, TrainingArguments
import datetime
import pandas_utils
import copy
import random
import logging
import create_occurrence_buckets
from competition_models import make_model, CompetitionTrainer, CompetitionModel
import json
import sklearn.metrics
import numpy as np

logger = logging.getLogger(__file__)


def parse_date(d: str) -> datetime.datetime:
    return datetime.datetime.strptime(d, "%Y-%m-%d")


@dataclass
class CompetitionArgs:
    pretrained_path: str = field(
        metadata={"help": "Pretrained model path"},
    )

    ref: str = field(
        metadata={"help": "Reference file"}
    )

    model_type: str = field(
        default="EncoderDecoder",
        metadata={"help": "Whether the model is an encoder or decoder type model"},
    )

    precomputed_train_pairings: Optional[str] = field(
        default=None,
        metadata={"help": "Precomputed train pairings files"}
    )

    precomputed_val_pairings: Optional[str] = field(
        default=None,
        metadata={"help": "Precomputed val pairings files"}
    )

    attn_lr_deboost: Optional[float] = field(
        default=1.0,
        metadata={"help": "By how much to scale down the learning rate of the attention layers"}
    )

    referee_type: Optional[str] = field(
        default="simple",
        metadata={"help": "Type of referee to be used: [simple, triplet, binary]"}
    )

    pooler_type: Optional[str] = field(
        default="mean",
        metadata={"help": "Type of pooler to be used: [mean, cls, max]"}
    )

    prediction_type: Optional[str] = field(
        default="weight",
        metadata={"help": "Type of prediction to make: [weight, binary]"}
    )

    max_train_set_size: Optional[int] = field(
        default=None,
        metadata={"help": "Restrict training to a subset"}
    )

    max_eval_set_size: Optional[int] = field(
        default=None,
        metadata={"help": "Restrict validation to a subset"}
    )

    prediction_output: Optional[str] = field(
        default=None,
        metadata={"help": "Optional output file for predictions"}
    )

    train_min_max_occurrence: Optional[int] = field(
        default=None,
        metadata={
            "help": "Out of each train pair what is the minimum value of the max occurrence"
        }
    )

    train_min_min_occurrence: Optional[int] = field(
        default=None,
        metadata={
            "help": "Out of each train pair what is the minimum value of the minimum occurrence"
        }
    )

    val_min_max_occurrence: Optional[int] = field(
        default=None,
        metadata={
            "help": "Out of each val pair what is the minimum value of the max occurrence"
        }
    )

    val_min_min_occurrence: Optional[int] = field(
        default=None,
        metadata={
            "help": "Out of each val pair what is the minimum value of the minimum occurrence"
        }
    )

    n_leading_week_counts: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of leading weekly occurrences to use in classifier"
        }
    )


def make_optimizer(
    model: CompetitionModel,
    training_args: TrainingArguments,
    competition_args: CompetitionArgs,
) -> torch.optim.AdamW:
    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
        "weight_decay": training_args.weight_decay,
    }
    parameter_groups = []
    parameter_groups.extend([
        {"params": model.core_model.parameters(), "lr": training_args.learning_rate * competition_args.attn_lr_deboost},
        {"params": model.pooler.parameters(), "lr": training_args.learning_rate},
        {"params": model.referee.parameters(), "lr": training_args.learning_rate},
    ])
    return torch.optim.AdamW(parameter_groups, **optimizer_kwargs)


def make_data_precomputed(
    train_pairings: str,
    val_pairings: str,
    ref: str,
    train_min_max_occurrence: Optional[int] = None,
    train_min_min_occurrence: Optional[int] = None,
    val_min_max_occurrence: Optional[int] = None,
    val_min_min_occurrence: Optional[int] = None,
    n_leading_week_counts: Optional[int] = None,
) -> tuple:
    return_data = []

    with open(ref, "r") as fhandle:
        ref = fhandle.read().strip()

    min_max_occurrence = {"train": train_min_max_occurrence, "val": val_min_max_occurrence}
    min_min_occurrence = {"train": train_min_min_occurrence, "val": val_min_min_occurrence}
    files = {"train": train_pairings, "val": val_pairings}

    for name in ["train", "val"]:
        with open(files[name], "r") as fhandle:
            pairings = [create_occurrence_buckets.SeqPair(**json.loads(item)) for item in fhandle]

        logging.info(f"Number of examples for {name}={len(pairings)}")

        return_data.append(sequence_feature_data_utils.PairwiseBucketizedDataset(
            pairings,
            ref=ref,
            min_max_occurrence=min_max_occurrence[name],
            min_min_occurrence=min_min_occurrence[name],
            n_leading_week_counts=n_leading_week_counts,
        ))

    logger.info(f"Theoretical minimum val loss={return_data[1].get_theoretical_min_loss()}")

    return tuple(return_data)


def make_data(competition_args: CompetitionArgs) -> tuple:
    logger.info("Initializing pairings from precomputed files")
    return make_data_precomputed(
        competition_args.precomputed_train_pairings,
        competition_args.precomputed_val_pairings,
        competition_args.ref,
        train_min_max_occurrence=competition_args.train_min_max_occurrence,
        train_min_min_occurrence=competition_args.train_min_min_occurrence,
        val_min_max_occurrence=competition_args.val_min_max_occurrence,
        val_min_min_occurrence=competition_args.val_min_min_occurrence,
        n_leading_week_counts=competition_args.n_leading_week_counts,
    )


def collate_function_wrapper(batch: list) -> dict:
    res = sequence_feature_data_utils.collate_function(batch)
    return {"seq_a_tokens": res[0], "seq_b_tokens": res[1], "labels": res[2]}


def collate_function_wrapper_with_leading_weeks(batch: list) -> dict:
    batch_tensors_labels = []
    batch_leading_weeks = []

    for item in batch:
        batch_tensors_labels.append((item[0], item[-1]))
        batch_leading_weeks.append(item[1])

    tensors_labels_collated = sequence_feature_data_utils.collate_function(batch_tensors_labels)

    return {
        "seq_a_tokens": tensors_labels_collated[0],
        "seq_b_tokens": tensors_labels_collated[1],
        "labels": tensors_labels_collated[2],
        "seq_a_leading_weeks": torch.Tensor([x[0] for x in batch_leading_weeks]),
        "seq_b_leading_weeks": torch.Tensor([x[1] for x in batch_leading_weeks]),
    }


def get_discretized_label(probs: np.ndarray, ranges: list):
    new_labels = np.zeros(shape=probs.shape, dtype=np.int32)

    for i, r in enumerate(ranges):
        new_labels = np.where(np.logical_and(probs >= r[0], probs < r[1]), i, new_labels)

    return new_labels


def get_keep_range(probs: list, ignore_range: list):
    return [i for i, p in enumerate(probs) if not(ignore_range[0] <= p < ignore_range[1])]


def calculate_accuracy(
    logits,
    labels,
    ranges: list = [[0, 0.5], [0.5, 1]],
) -> tuple:
    y_true = get_discretized_label(labels.numpy(), ranges)
    p1 = torch.sigmoid(logits)
    y_predict = get_discretized_label(p1.numpy(), ranges)

    f1_micro = sklearn.metrics.f1_score(y_true, y_predict, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_predict, average="macro")
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_predict)

    return f1_micro, f1_macro, confusion_matrix


def calculate_accuracy_top(
    results,
    ranges: list = [[0, 0.5], [0.5, 1]]
) -> None:
    logits = torch.Tensor(results.predictions)
    labels = torch.Tensor(results.label_ids)
    idx_labels_zero_or_one = torch.arange(0, labels.shape[0]).masked_select(
        torch.logical_or(labels == 0, labels == 1)
    )
    idx_labels_not_zero_or_one = torch.arange(0, labels.shape[0]).masked_select(
        torch.logical_and(labels != 0, labels != 1)
    )

    if idx_labels_zero_or_one.numel() > 0:
        logits_fake = logits[idx_labels_zero_or_one]
        labels_fake = labels[idx_labels_zero_or_one]
        measure_fake = calculate_accuracy(logits_fake, labels_fake, ranges)

        logger.info(
            f"F-1 score [real-vs-fake]: micro = {measure_fake[0]}, "
            f"macro = {measure_fake[1]}; confusion_matrix = {measure_fake[2]}"
        )

    if idx_labels_not_zero_or_one.numel() > 0:
        logits_real = logits[idx_labels_not_zero_or_one]
        labels_real = labels[idx_labels_not_zero_or_one]
        measure_real = calculate_accuracy(logits_real, labels_real, ranges)

        logger.info(
            f"F-1 score [real-vs-real]: micro = {measure_real[0]}, "
            f"macro = {measure_real[1]}; confusion_matrix = {measure_real[2]}"
        )

    measure_overall = calculate_accuracy(logits, labels, ranges)

    logger.info(
        f"F-1 score [overall]: micro = {measure_overall[0]}, "
        f"macro = {measure_overall[1]}; confusion_matrix = {measure_overall[2]}"
    )


def main():
    parser = HfArgumentParser((CompetitionArgs, TrainingArguments))
    competition_args, training_args = parser.parse_args_into_dataclasses()
    random.seed(training_args.seed)
    log_level = training_args.get_process_log_level()
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s:%(message)s")
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info("Initializing data, models and trainer")
    train_dataset, val_dataset = make_data(competition_args)

    if competition_args.max_train_set_size:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(competition_args.max_train_set_size))

    if competition_args.max_eval_set_size:
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(competition_args.max_eval_set_size))

    model = make_model(
        competition_args.pretrained_path,
        referee_type=competition_args.referee_type,
        prediction_type=competition_args.prediction_type,
        pooler_type=competition_args.pooler_type,
        model_type=competition_args.model_type,
        n_leading_weeks=competition_args.n_leading_week_counts,
    )

    collator = collate_function_wrapper

    if competition_args.n_leading_week_counts:
        collator = collate_function_wrapper_with_leading_weeks

    trainer = CompetitionTrainer(
        model,
        training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(make_optimizer(model, training_args, competition_args), None),
    )

    if training_args.do_predict and training_args.do_train and training_args.num_train_epochs > 0:
        logger.info("*** Pretrained Predictions ***")
        results = trainer.predict(test_dataset=val_dataset)
        calculate_accuracy(results)

    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["val_samples"] = len(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(test_dataset=val_dataset)
        calculate_accuracy_top(results)
        if competition_args.prediction_output:
            np.savez(
                competition_args.prediction_output,
                predictions=results.predictions,
                labels=results.label_ids
            )


if __name__ == "__main__":
    main()
