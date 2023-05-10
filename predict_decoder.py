# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import models
import data_processing
from typing import List, Union
from itertools import takewhile
import tqdm
import logging
import argparse
from prediction_tools import add_sampler_parameters
import torch
import json
import os
from dataclasses import dataclass
from transformers import TrainingArguments
from train_quark_finetune import ModelArguments, make_model
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer import TRAINING_ARGS_NAME
from utils import _DEFAULT_SPECIAL_TOKENS
from quark_finetune import QuarkModel
from transformers import BertLMHeadModel
from quark_finetune import calc_likelihoods_top
import transformers

logger = logging.getLogger(__file__)

_MAPPER = data_processing.Tokenizer().mapper
_EOS_TOKEN = _MAPPER[_DEFAULT_SPECIAL_TOKENS.end_of_sequence]
_BOS_TOKEN = _MAPPER[_DEFAULT_SPECIAL_TOKENS.start_of_sequence]


def convert_tensor_to_seq(generated: torch.Tensor, reverse_mapper: dict, quark: bool = False) -> List[str]:
    results = []

    for g in generated.cpu().tolist():
        res = ""
        valid = False

        iterable = iter(g[1:]) if not quark else iter(g[2:])

        try:
            for i in iterable:
                aa = reverse_mapper[i]

                if aa == "[SEP]":
                    aa = "*"

                res += aa

                if aa == "*":
                    valid = True
                    break
        except KeyError:
            continue

        results.append((res, valid))

    return results


class QuarkPredictor(torch.nn.Module):
    def __init__(self, quark_model: QuarkModel, quantile_token: int):
        super().__init__()
        self.train_model = quark_model.train_model
        self.quantile_token = quantile_token

    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except StopIteration:
            return False

    def get_device(self):
        return next(self.parameters()).get_device()

    def generate(self, bos_token_id: int, eos_token_id: int, **gen_kwargs):
        bos = torch.LongTensor([bos_token_id, self.quantile_token])[None]

        if self.is_cuda:
            bos = bos.to(self.get_device())

        return self.train_model.generate(input_ids=bos, eos_token_id=eos_token_id, **gen_kwargs)

    def forward(self, *args, **kwargs):
        return self.train_model(*args, **kwargs)


def load_quark_model(pth: str):
    filename = os.path.join(pth, TRAINING_ARGS_NAME)
    training_args = torch.load(filename)
    model_args = training_args.model_args
    quark_model = make_model(model_args)
    weights = torch.load(os.path.join(pth, WEIGHTS_NAME), map_location="cpu")
    quark_model.load_state_dict(weights)
    quantile_offset = training_args.quantile_offset
    n_quantiles = len(training_args.quantiles)
    quark_predictor = QuarkPredictor(quark_model, n_quantiles + quantile_offset - 1)
    return quark_predictor


def is_cuda(model: torch.nn.Module):
    return next(model.parameters()).is_cuda


def get_device(model: torch.nn.Module):
    return next(model.parameters()).get_device()


def main(args):
    if args.quark_model:
        model = load_quark_model(args.checkpoint)
    else:
        model = models.from_pretrained(args.checkpoint, "Decoder")

    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    mapper = data_processing.Tokenizer().mapper
    reverse_mapper = {}

    for k, v in mapper.items():
        if v not in reverse_mapper:
            reverse_mapper[v] = k

    gen_kwargs = {a[4:]: getattr(args, a) for a in vars(args) if a[:4] == "gen_"}
    all_results = []
    all_generations = []

    for i in tqdm.tqdm(range(args.num_batches), desc="Generating sequences"):
        res = model.generate(bos_token_id=_BOS_TOKEN, eos_token_id=_EOS_TOKEN, **gen_kwargs)
        all_generations.append(res.cpu())

    max_length = max(x.shape[1] for x in all_generations)
    all_generations = [
        torch.nn.functional.pad(x, pad=(0, max_length - x.shape[1])) for x in all_generations]
    all_generations = torch.cat(all_generations, dim=0)

    all_likelihoods = []

    for res in tqdm.tqdm(
        torch.split(
            all_generations,
            split_size_or_sections=gen_kwargs["num_return_sequences"],
            dim=0
        ),
        desc="Calculating likelihoods",
    ):
        with torch.no_grad():
            """
            Note: Transformers library automatically sets causal language modeling mask
            when is_decoder is True for Bert. This was clarified in the following issue:
            https://github.com/huggingface/transformers/issues/12704

            The corresponding code lines are here: https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/modeling_utils.py#L273
            From def forward, we see the following trace:
            https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/bert/modeling_bert.py#L969
            However, this function is not created in BertModel, but in the inheritance heirarchy. That hierarchy
            has BertPreTrainedModel -> PretrainedModel which comes from modeling_utils which inherits
            (multiply) from ModuleUtilsMixin which contains the lines referred to in the github issue.
            """
            if is_cuda(model):
                res = res.to(get_device(model))

            likelihoods = calc_likelihoods_top(
                model,
                res,
                is_quark_model=args.quark_model,
                eos_token=_EOS_TOKEN,
            )

            all_likelihoods.extend(likelihoods.cpu().tolist())

    for (s, v), l in tqdm.tqdm(
        zip(convert_tensor_to_seq(all_generations, reverse_mapper, quark=args.quark_model), all_likelihoods),
        desc="Formatting generated sequences"
    ):
        if v:
            all_results.append({"seq": s, "ll": l})

    with open(args.output_prefix + ".json", "w") as fhandle:
        for i, item in enumerate(all_results):
            fhandle.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequences from decoder-only model")

    parser.add_argument("--checkpoint", help="Checkpoint to use for generation", required=True)
    parser.add_argument("--output_prefix", help="Prefix of output file", required=True)
    parser.add_argument("--num_batches", help="Number of batches of generations to run", type=int, required=True)
    parser.add_argument("--quark_model", help="Indicate that we want quark model generation", action="store_true", default=False)
    parser.add_argument("--seed", help="Seed for generation", default=None, type=int)

    add_sampler_parameters(parser)

    args = parser.parse_args()

    if args.seed:
        transformers.set_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    main(args)
