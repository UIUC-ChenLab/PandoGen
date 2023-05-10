# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import torch
import competition_models
import models
import transformers
from transformers import HfArgumentParser, TrainingArguments, Trainer
from dataclasses import dataclass, field
from utils import get_full_sequence
from data_processing import Tokenizer
from typing import Optional
import numpy as np
import csv
import logging
import os
import random
import json
from utils import _DEFAULT_SPECIAL_TOKENS, SpecialTokens


@dataclass
class CompetitionPredictionArgs:
    checkpoint_path: str = field(
        metadata={"help": "Path to the checkpoint to use"}
    )

    pretrained_path: str = field(
        metadata={"help": "Original pretrained encoder-decoder path"}
    )
    
    target_sequences: str = field(
        metadata={"help": "Target sequences for which to compute"}
    )

    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Output filename into which prediction results are to be written"}
    )

    model_type: str = field(
        default="EncoderDecoder",
        metadata={"help": "Whether the model is an encoder or decoder type model"},
    )

    sars_cov2_ref: Optional[str] = field(
        default=None,
        metadata={"help": "SARS-CoV2 reference sequence"}
    )

    reference_sequences: Optional[str] = field(
        default=None,
        metadata={"help": "List of reference sequences"}
    )

    referee_type: Optional[str] = field(
        default="binary",
        metadata={"help": "Referee type"}
    )

    pooler_type: Optional[str] = field(
        default="mean",
        metadata={"help": "Pooler type"}
    )

    embedding_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size to produce embeddings"}
    )

    predict_batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "Batch size to produce predictions"}
    )

    n_leading_weeks: Optional[int] = field(
        default=None,
        metadata={"help": "Number of leading weeks to use in predictions"}
    )

    workdir: Optional[str] = field(
        default=None,
        metadata={"help": "A working directory for scripts calling this one"},
    )


def make_models(args: CompetitionPredictionArgs) -> tuple:
    competition_model = competition_models.make_model(
        args.pretrained_path,
        args.referee_type,
        prediction_type="weight",
        pooler_type=args.pooler_type,
        model_type=args.model_type,
        n_leading_weeks=args.n_leading_weeks,
    )
    weights = torch.load(
        os.path.join(args.checkpoint_path, "pytorch_model.bin"),
        map_location="cpu",
    )
    competition_model.load_state_dict(weights)
    embedder = competition_models.Embedder(competition_model)
    relative_predictor = competition_models.Predictor(competition_model)
    potential_predictor = competition_models.PredictPotential(competition_model)
    embedder.eval()
    relative_predictor.eval()
    potential_predictor.eval()
    return (embedder, relative_predictor, potential_predictor)


class SingleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequences: list,
        ref: Optional[str] = None,
        special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS
    ):
        super().__init__()
        self.ref = ref
        self.sequences = sequences
        self.mapper = Tokenizer().mapper
        self.special_tokens = special_tokens

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        item = self.sequences[idx]

        if not self.ref:
            full_seq = item
        else:
            full_seq = get_full_sequence(item, self.ref)

        full_seq = [self.special_tokens.start_of_sequence] + list(full_seq)
        mapped = [self.mapper[i] for i in full_seq]

        return torch.LongTensor(mapped)


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        embeddings: torch.Tensor,
        leading_weeks: Optional[list] = None,
        n_leading_weeks: Optional[int] = None,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.leading_weeks = leading_weeks
        self.n_leading_weeks = n_leading_weeks

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.leading_weeks and self.n_leading_weeks:
            leading_weeks = torch.Tensor(self.leading_weeks[idx])[: self.n_leading_weeks]
        else:
            leading_weeks = None

        return torch.Tensor(self.embeddings[idx]), leading_weeks


def zeropad_tokens(batch: list) -> dict:
    max_length = max(t.shape[0] for t in batch)
    input_ids = torch.zeros(len(batch), max_length).long()
    attention_mask = torch.zeros(len(batch), max_length).byte()
    for i, b in enumerate(batch):
        input_ids[i, :b.shape[0]] = b
        attention_mask[i, :b.shape[0]] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def simple_collator(batch: list) -> dict:
    return {"sequence_tokens": zeropad_tokens(batch)}


def embeddings_collator(batch: list) -> dict:
    embeds, leading_weeks = tuple(zip(*batch))
    embeds_stacked = torch.stack(embeds, dim=0)
    return_dict = {"embeddings": embeds_stacked}

    if leading_weeks[0] is not None:
        leading_weeks_tensor = torch.stack(leading_weeks, dim=0)
        return_dict["leading_weeks"] = leading_weeks_tensor

    return return_dict


def get_embeddings(
    sequences: list,
    ref: str,
    model: competition_models.Embedder,
    predict_batch_size: int = 32,
    workdir: Optional[str] = None,
):
    """
    We use Transformers Trainer to obtain prediction instead of writing a loop ourselves

    Note that for evaluation and test datasets, it uses SequentialSampler, so
    the embeddings are simply in the order the input data is arranged in:
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L2278
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L740
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L783
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L696
    """
    if not workdir:
        workdir = "/tmp"

    dataset = SingleDataset(sequences, ref)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=workdir,
            per_device_eval_batch_size=predict_batch_size,
        ),
        data_collator=simple_collator,
    )

    """
    Note that loss is not computed when labels aren't present. The model outputs get routed via
    logits to predictions:
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L2597
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L2884
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L2893
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L2910
    """
    res = trainer.predict(test_dataset=dataset)
    return res.predictions


def get_predictions_single(
    embeddings: torch.Tensor,
    model: competition_models.PredictPotential,
    predict_batch_size: int,
    workdir: Optional[str] = None,
    leading_weeks: Optional[list] = None,
    n_leading_weeks: Optional[int] = None,
) -> torch.Tensor:
    """
    Get individual predictions for each sequence
    """
    if not workdir:
        workdir = "/tmp"

    dset = EmbeddingDataset(
        embeddings, leading_weeks=leading_weeks, n_leading_weeks=n_leading_weeks)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=workdir,
            per_device_eval_batch_size=predict_batch_size,
        ),
        data_collator=embeddings_collator,
    )

    res = trainer.predict(test_dataset=dset)

    return res.predictions


def read_data(filename: str, n_leading_weeks: Optional[int] = None) -> tuple:
    sequences = []
    leading_weeks = []

    with open(filename, "r") as fhandle:
        for line in fhandle:
            item = json.loads(line)

            if "stop" in item["seq"] or "[" in item["seq"] or "]" in item["seq"]:
                continue

            sequences.append(item["seq"])
            leading_week_item = item.get("weekly_counts", None)
            if n_leading_weeks:
                if len(leading_week_item) < n_leading_weeks:
                    leading_week_item.extend([0] * (n_leading_weeks - len(leading_week_item)))
            leading_weeks.append(leading_week_item)

    return sequences, leading_weeks


def potential_prediction_top(args: CompetitionPredictionArgs) -> dict:
    embedder, _, potential_predictor = make_models(args)
    sequences, leading_weeks = read_data(args.target_sequences, args.n_leading_weeks)

    # Sanity-check to make sure there is no unfounded bias due to
    # a specific order in the input
    randindices = list(range(len(sequences)))
    random.shuffle(randindices)
    sequences = [sequences[i] for i in randindices]
    leading_weeks = [leading_weeks[i] for i in randindices]

    if args.sars_cov2_ref:
        with open(args.sars_cov2_ref, "r") as fhandle:
            ref = fhandle.read().strip()
    else:
        ref = None

    embeddings = get_embeddings(
        sequences,
        ref,
        embedder,
        args.embedding_batch_size,
    )

    potentials = get_predictions_single(
        embeddings,
        potential_predictor,
        predict_batch_size=args.predict_batch_size,
        leading_weeks=leading_weeks,
        n_leading_weeks=args.n_leading_weeks,
    )

    return dict(zip(sequences, potentials[:, 0].tolist()))


def main(args: CompetitionPredictionArgs):
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s:%(message)s")
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if args.reference_sequences:
        raise NotImplementedError("Pairwise scoring has not been implemented")
    else:
        res = potential_prediction_top(args)

        with open(args.output_path, "w", newline="") as fhandle:
            writer = csv.DictWriter(fhandle, fieldnames=["seq0", "potential"])
            writer.writeheader()
            for seq, p in res.items():
                writer.writerow({"seq0": seq, "potential": p})


if __name__ == "__main__":
    parser = HfArgumentParser((CompetitionPredictionArgs, ))
    args,  = parser.parse_args_into_dataclasses()
    main(args)
