# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import torch
import models
from transformers import EncoderDecoderModel
from typing import List, Tuple, Callable, Union, Optional
from data_processing import split_sequence, Tokenizer
from utils import (
    SpecialTokens,
    _DEFAULT_SPECIAL_TOKENS,
    is_cuda,
    fasta_serial_reader,
    mutation_positions_in_seq
)
import logging
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from itertools import takewhile
from Bio import pairwise2
import argparse
import json
import pysam
import tqdm
import math
from functools import partial, lru_cache
from enum import Enum, auto

logger = logging.getLogger(__file__)

_MATCH_SCORE = 1
_MISMATCH_PENALTY = -1
_GAP_OPEN_PENALTY = -4
_GAP_EXTEND_PENALTY = -0.1


class CompareType(Enum):
    LL = auto()
    LL_RATIO = auto()
    PRE_EXISTING = auto()


@dataclass
class Stimulus:
    seq: str
    mask_position: str
    mask_length: str
    encoder_seq: str
    decoder_seq: str


@dataclass(order=True)
class MutationResult:
    parent: str = field(compare=False)
    child: str = field(compare=False)
    metric: Optional[float] = field(compare=True, default=-float("inf"))
    ll: Optional[float] = field(compare=False, default=-float("inf"))
    ll_ratio: Optional[float] = field(compare=False, default=-float("inf"))
    compare_type: CompareType = field(compare=False, default=CompareType.LL)
    parent_mutation_repr: Optional[str] = field(compare=False, default=None)
    child_mutation_repr: Optional[str] = field(compare=False, default=None)
    potential: Optional[float] = field(compare=False, default=None)

    def __post_init__(self):
        if self.compare_type is CompareType.LL:
            self.metric = float(self.ll)
        elif self.compare_type is CompareType.LL_RATIO:
            self.metric = float(self.ll_ratio)
        elif self.compare_type is CompareType.PRE_EXISTING:
            self.metric = float(self.metric)
        else:
            raise AttributeError(f"Bad attribute compare_type = {self.compare_type}")

        if type(self.parent_mutation_repr) is list:
            self.parent_mutation_repr = ",".join(self.parent_mutation_repr)

        if type(self.child_mutation_repr) is list:
            self.child_mutation_repr = ",".join(self.child_mutation_repr)

    def as_json_dict(self):
        d = asdict(self)
        del d["compare_type"]
        return d


@dataclass
class Predictions:
    seq: str
    mask_position: str
    mask_length: str
    encoder_seq: str
    decoder_seq: str
    predicted_sequences: Union[torch.LongTensor, List[str]]
    predicted_lls: torch.Tensor
    predicted_ll_ratios: torch.Tensor
    reverse_mapper: dict
    differing_sequences: Optional[List[str]] = None
    mutation_sequences: Optional[List[str]] = None
    differing_lls: Optional[torch.Tensor] = None
    differing_ll_ratios: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Convert tensor to decoded sequence
        predicted_sequences = []
        for seq in self.predicted_sequences.cpu().tolist():
            seq_str = list(takewhile(
                lambda x: x != "[MASK]", 
                map(lambda x: self.reverse_mapper[x], seq)
            ))
            predicted_sequences.append(seq_str)

        self.predicted_sequences = predicted_sequences

        # Collect only differing sequences
        # Remove the last item as decoder_seq has a "[MASK]" token at the end
        # that the predicted sequence doesn't have (because we removed [MASK] above)
        differing_indices = [
            i for i, x in enumerate(self.predicted_sequences) if (
                (x != self.decoder_seq[: -1]) and ("[SEP]" not in x) and ("[CLS]" not in x))]
        self.differing_sequences = [self.predicted_sequences[i] for i in differing_indices]
        self.differing_lls = self.differing_lls if not differing_indices else \
            self.predicted_lls[differing_indices]
        self.differing_ll_ratios = self.differing_ll_ratios if not differing_indices else \
            self.predicted_ll_ratios[differing_indices]

    def _get_full_sequence(self, segment: list) -> str:
        return self.seq[: self.mask_position] + "".join(segment) + \
            self.seq[self.mask_position + self.mask_length: ]

    def get_mutation_seq(self, ref: str) -> List[str]:
        if self.mutation_sequences is None:
            mutation_sequences = []

            for segment in self.differing_sequences:
                full_sequence = self._get_full_sequence(segment)
                mutation_sequences.append(",".join(get_mutations(ref, full_sequence)))

            self.mutation_sequences = mutation_sequences

        return self.mutation_sequences

    @property
    def json(self):
        return {
            "seq": self.seq,
            "mask_position": self.mask_position,
            "mask_length": self.mask_length,
            "encoder_seq": self.encoder_seq,
            "decoder_seq": self.decoder_seq,
            "differing_sequences": self.differing_sequences,
            "mutation_sequences": self.mutation_sequences,
            "differing_lls": self.differing_lls.cpu().tolist(),
        }


@lru_cache(None)
def get_mutations(ref: str, seq: str) -> List[str]:
    if ref == seq:
        return ""

    alignment = pairwise2.align.globalms(
        ref, seq, _MATCH_SCORE, _MISMATCH_PENALTY, _GAP_OPEN_PENALTY, _GAP_EXTEND_PENALTY)[0]

    ref_ptr = 0
    mutations = []
    state = None

    for r, s in zip(alignment.seqA, alignment.seqB):
        s = s if s != "," else "-"
        if r == s:
            state = "Match"
            if r != "-":
                ref_ptr += 1
        elif r != "-" and s == "-":
            state = "Deletion"
            mutations.append(f"{r}{ref_ptr + 1}del")
            ref_ptr += 1
        elif r == "-" and s != "-":
            # Insertions are appended to the right-side
            # of the AA at the position. Hence, this is the
            # correct way to do it
            if state == "Insertion":
                mutations[-1] += s
            else:
                mutations.append(f"ins{ref_ptr}{s}")
            state = "Insertion"
        else:
            state = "Mismatch"
            mutations.append(f"{r}{ref_ptr + 1}{s}")
            ref_ptr += 1

    return mutations


def split_locations(
    window: tuple,
    frozen_locations: list = list()
):
    mask_positions = []
    remaining_segment = window

    if any(window[0] <= f < window[1] for f in frozen_locations):
        for f in frozen_locations:
            if remaining_segment[0] <= f < remaining_segment[1]:
                xlice = (remaining_segment[0], f)
                if xlice[1] - xlice[0] > 0:
                    mask_positions.append(xlice)
                remaining_segment = (f + 1, remaining_segment[1])

    if remaining_segment[1] - remaining_segment[0] > 0:
        mask_positions.append(remaining_segment)

    return mask_positions


def prepare_stimulii(
    seq: str,
    window_size: int = 64,
    step_size: int = 32,
    frozen_locations: list = list(),
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
) -> List[Stimulus]:
    """
    Prepare all edit stimuli for a given sequence.
    """
    splits = []

    # Do not include the terminus
    for start_pos in range(0, len(seq), step_size):
        window = [start_pos, min(start_pos + window_size, len(seq) - 1)]

        if window[1] - window[0] < 1:
            continue

        for mask_spec in split_locations(window, frozen_locations):
            encoder_seq, decoder_seq = split_sequence(
                sequence=seq,
                mask_start=mask_spec[0],
                mask_length=mask_spec[1] - mask_spec[0],
                special_tokens=special_tokens,
            )
            splits.append(
                Stimulus(
                    seq,
                    mask_spec[0],
                    mask_spec[1] - mask_spec[0],
                    encoder_seq,
                    decoder_seq,
                )
            )
    return splits


def calc_likelihoods(
    decoded: torch.LongTensor,
    end_token_value: int,
    logits: torch.Tensor,
):
    # Create a mask - we need to include the mask location, so we right-shift the original
    mask = torch.nn.functional.pad(
        torch.cumprod(torch.ne(decoded, end_token_value), dim=1),
        pad=[1, 0, 0, 0], value=1)[:, :-1]

    # Check whether seq termminal is found somewhere in the sequence
    # If not, that generation is invalid
    found_seq_terminal = torch.sum(decoded == end_token_value, dim=1) > 0

    # Create distribution
    dist = torch.distributions.categorical.Categorical(logits=logits)

    # Calculate ll
    ll = torch.sum(dist.log_prob(decoded).masked_fill(mask == 0, 0), dim=1)

    return torch.where(found_seq_terminal, ll, torch.zeros_like(ll) - 1e5)


def zeropad(sequences: List[Stimulus], mapper: dict) -> tuple:
    max_encoder_length = max(len(s.encoder_seq) for s in sequences)
    max_decoder_length = max(len(s.decoder_seq) for s in sequences)
    input_ids = torch.zeros(len(sequences), max_encoder_length).long()
    attention_mask = torch.zeros(len(sequences), max_encoder_length).byte()
    decoder_ids = torch.zeros(len(sequences), max_decoder_length).long() - 100
    for i, s in enumerate(sequences):
        encoded = torch.LongTensor([mapper[i] for i in s.encoder_seq])
        decoded = torch.LongTensor([mapper[i] for i in s.decoder_seq])
        input_ids[i, :encoded.shape[0]] = encoded
        attention_mask[i, :encoded.shape[0]] = 1
        decoder_ids[i, :decoded.shape[0]] = decoded
    return input_ids, attention_mask, decoder_ids


def batch_generate(
    model: EncoderDecoderModel,
    batch_size: int,
    input_ids: torch.LongTensor,
    attention_mask: torch.ByteTensor,
    **kwargs,
):
    tokenizer = Tokenizer()
    n_chunks = math.ceil(input_ids.shape[0] / batch_size)
    with torch.no_grad():
        generated = []
        for i, a in zip(
            torch.chunk(input_ids, n_chunks, dim=0),
            torch.chunk(attention_mask, n_chunks, dim=0),
        ):
            gen = model.generate(
                input_ids=i,
                attention_mask=a,
                eos_token_id=tokenizer.mapper["[MASK]"],
                pad_token_id=tokenizer.mapper["[MASK]"],
                **kwargs,
            ).cpu()
            generated.append(gen)

    max_length = max(g.shape[1] for g in generated)
    pad_value = tokenizer.mapper["[MASK]"]

    return torch.cat(
        [torch.nn.functional.pad(
            g, pad=[0, max_length - g.shape[1]], value=pad_value) for g in generated], dim=0)


def chunk_function(items: Optional[torch.Tensor] = None, n_chunks: int = -1) -> list:
    assert(n_chunks > 0)
    if items is None:
        return [None] * n_chunks
    else:
        return torch.chunk(items, chunks=n_chunks, dim=0)


def batch_calculate(
    model: EncoderDecoderModel,
    input_ids: torch.LongTensor,
    attention_mask: torch.ByteTensor,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    batch_size: int = 16,
):
    """
    Note: According to https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L508
    If labels is provided it is automatically right shifted by 1, and the last position is clipped,
    so the decoder inputs produced this way is the same length as the labels. The first position
    is ensured to be the [CLS] token (based on the config).

    If decoder_input_ids are given, it needs to have its first position be using the [CLS] token.
    Generated decoder tokens have this property, so its not a problem.
    """

    """
    Note: In EncoderDecoderModel, attention_mask refers to encoder attention mask. There is no
    need to pass decoder_attention_mask. See lines 60-67 in predict_decoder.py for reasoning.
    """
    if decoder_input_ids is None and labels is None:
        raise ValueError("Cannot have both decoder_input_ids and labels as None")

    n_chunks = math.ceil(input_ids.shape[0] / batch_size)
    logits = []
    chunker = partial(chunk_function, n_chunks=n_chunks)

    for i, a, d, l in zip(
        chunker(input_ids),
        chunker(attention_mask),
        chunker(decoder_input_ids),
        chunker(labels),
    ):
        logits.append(model(input_ids=i, attention_mask=a, decoder_input_ids=d, labels=l).logits)

    return torch.cat(logits, dim=0)


def predict(
    model: EncoderDecoderModel,
    stimulii: List[Stimulus],
    forward_mapper: dict,
    reverse_mapper: dict,
    batch_size_gen: int = 16,
    batch_size_ll: int = 16,
    compare_type: str = "ll",
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
    **gen_kwargs,
) -> List[Predictions]:
    """
    Predict potential mutations of a given set of stimuli, parse them and
    return continuations and their probabilities.
    """
    input_ids, attention_mask, decoder_ids = zeropad(stimulii, forward_mapper)

    if is_cuda(model):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    generated = batch_generate(
        model=model,
        batch_size=batch_size_gen,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
    num_return_sequences = generated.shape[0] // input_ids.shape[0]

    def expand_tensor_util(tensor: torch.Tensor) -> torch.Tensor:
        repeat_config = [1, num_return_sequences] + [1 for i in range(len(tensor.shape) - 1)]
        tensor = torch.flatten(
            torch.unsqueeze(tensor, dim=1).repeat(*repeat_config),
            start_dim=0, end_dim=1,
        )
        return tensor

    with torch.no_grad():
        cudify = lambda x: x.cuda() if is_cuda(model) else x
        logits = batch_calculate(
            model,
            input_ids=cudify(expand_tensor_util(input_ids)),
            attention_mask=cudify(expand_tensor_util(attention_mask)),
            decoder_input_ids=cudify(generated),
            batch_size=batch_size_ll,
        )
        ll = calc_likelihoods(
            cudify(generated[:, 1:]),
            forward_mapper[special_tokens.masked_segment_indicator],
            cudify(logits[:, :-1])
        )
        decoder_ids = cudify(decoder_ids)
        logits_wt = batch_calculate(
            model,
            input_ids=cudify(input_ids),
            attention_mask=cudify(attention_mask),
            labels=decoder_ids,
            batch_size=batch_size_ll,
        )
        dist_wt = torch.distributions.categorical.Categorical(logits=logits_wt)
        ll_wt = torch.sum(
            dist_wt.log_prob(
                decoder_ids.masked_fill(decoder_ids == -100, 0)
            ).masked_fill(decoder_ids == -100, 0),
            dim=1,
        )
        ll_ratio = ll - expand_tensor_util(ll_wt)

    generated_sep = torch.chunk(generated.cpu(), chunks=len(stimulii), dim=0)
    ll_sep = torch.chunk(ll.cpu(), chunks=len(stimulii), dim=0)
    ll_ratio_sep = torch.chunk(ll_ratio.cpu(), chunks=len(stimulii), dim=0)
    predictions = []

    for s, g, l, lr in zip(stimulii, generated_sep, ll_sep, ll_ratio_sep):
        predictions.append(
            Predictions(
                **asdict(s),
                predicted_sequences=g[:, 1:],
                predicted_lls=l,
                predicted_ll_ratios=lr,
                reverse_mapper=reverse_mapper,
            )
        )

    return predictions


def sort_predictions(
    predictions: List[Predictions],
    sequence_ll_dict: dict,
    sequence_ll_ratio_dict: dict,
    ref: Optional[str] = None,
    compare_type: CompareType = CompareType.LL,
) -> Predictions:
    """
    Find the best prediction
    """
    mutation_results = dict()

    for p in predictions:
        if p.differing_lls is not None:
            for i, (d, l, lr) in enumerate(
                zip(
                    p.differing_sequences,
                    p.differing_lls.cpu().tolist(),
                    p.differing_ll_ratios.tolist()
                )
            ):
                full_seq = p._get_full_sequence(d)

                if full_seq in mutation_results:
                    prev = mutation_results[full_seq]
                    add_seq = prev.ll < l
                else:
                    add_seq = True

                if add_seq:
                    child_mut_repr = get_mutations(ref, full_seq) if ref else None
                    parent_ll = sequence_ll_dict[p.seq]
                    parent_ll_ratio = sequence_ll_ratio_dict[p.seq]
                    sequence_ll_dict[full_seq] = max(
                        sequence_ll_dict.get(full_seq, -float("inf")), l + parent_ll)
                    sequence_ll_ratio_dict[full_seq] = max(
                        sequence_ll_ratio_dict.get(full_seq, -float("inf")), lr + parent_ll_ratio)
                    mutation_results[full_seq] = MutationResult(
                        parent=p.seq,
                        child=full_seq,
                        ll=sequence_ll_dict[full_seq],
                        ll_ratio=sequence_ll_ratio_dict[full_seq],
                        compare_type=compare_type,
                        child_mutation_repr=child_mut_repr,
                    )

    return sorted(mutation_results.values(), reverse=True)


def predict_evolutionary_path(
    seq: str,
    mutation_sequence: str,
    ref: str,
    model: EncoderDecoderModel,
    num_steps: int = 4,
    window_size: int = 64,
    step_size: int = 32,
    batch_size_gen: int = 16,
    batch_size_ll: int = 16,
    max_num_seq_per_generation: int = 4,
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
    exclusions: Optional[set] = None,
    freeze_mutation_locations: bool = False,
    compare_type: CompareType = CompareType.LL,
    **gen_kwargs,
) -> List[MutationResult]:
    """
    Predict a set of evolutionary steps in a given amino acid sequence
    """
    tokenizer = Tokenizer(special_tokens=special_tokens)
    forward_mapper = tokenizer.mapper
    reverse_mapper = dict()

    for k, v in forward_mapper.items():
        if v not in reverse_mapper:
            reverse_mapper[v] = k

    if mutation_sequence == "REF":
        mutation_sequence = ""

    sequence_set = [
        MutationResult(
            parent=ref,
            child=seq,
            ll=0,
            ll_ratio=0,
            child_mutation_repr=mutation_sequence,
        )
    ]
    generations = [sequence_set]

    sequence_likelihoods = {sequence_set[0].child: sequence_set[0].ll}
    sequence_likelihood_ratios = {sequence_set[0].child: sequence_set[0].ll_ratio}

    for i in range(num_steps):
        logger.info(f"Performing prediction step {i}")
        logger.info(f"Collecting sequence splits")
        stimulii = []

        for s in sequence_set:
            frozen_locations = []

            if freeze_mutation_locations:
                for a, b in mutation_positions_in_seq(s.child_mutation_repr):
                    frozen_locations.extend(range(a, b))

            stimulii.extend(
                prepare_stimulii(
                    s.child,
                    window_size=window_size,
                    step_size=step_size,
                    frozen_locations=frozen_locations
                )
            )

        logger.info(f"Collected {len(stimulii)} splits, simulating putative mutations ... ")

        all_predictions = predict(
            model,
            stimulii,
            forward_mapper,
            reverse_mapper,
            batch_size_gen=batch_size_gen,
            batch_size_ll=batch_size_ll,
            special_tokens=special_tokens,
            **gen_kwargs,
        )

        logger.info(f"Sorting predictions")

        sequence_set = sort_predictions(
            all_predictions,
            sequence_ll_dict=sequence_likelihoods,
            sequence_ll_ratio_dict=sequence_likelihood_ratios,
            ref=ref if freeze_mutation_locations else None,
            compare_type=compare_type,
        )

        if exclusions:
            logger.info("Removing exclusions")
            sequence_set = [s for s in sequence_set if s.child not in exclusions]

        sequence_set = sequence_set[:max_num_seq_per_generation]

        if not sequence_set:
            logger.info(
                f"Found no valid prediction in step {i}. Terminating search."
            )
            break

        generations.append(sequence_set)

    for seq_set in generations:
        for s in seq_set:
            if not s.child_mutation_repr:
                s.child_mutation_repr = ",".join(get_mutations(ref, s.child))
            s.parent_mutation_repr = ",".join(get_mutations(ref, s.parent))

    return generations


def main(args):
    with open(args.ref) as fhandle:
        ref = fhandle.read().strip()

    exclusions = set()

    if args.exclusions:
        for exclude_file in args.exclusions.split(","):
            for item in fasta_serial_reader(exclude_file):
                exclusions.add(item.sequence)

    model = models.from_pretrained(args.model_checkpoint)
    model.eval()

    if args.cuda:
        model.cuda()

    gen_kwargs = {a[4:]: getattr(args, a) for a in vars(args) if a[:4] == "gen_"}

    with pysam.FastaFile(args.seq) as fhandle, \
        open(f"{args.output_prefix}.json", "w") as whandle:
        logger.info(f"Found {len(fhandle.references)} sequences. Computing.")
        for r in tqdm.tqdm(fhandle.references, desc="Sequence progress"):
            seq = fhandle.fetch(r)
            generations = predict_evolutionary_path(
                seq,
                r.replace("_", ","),
                ref,
                model=model,
                num_steps=args.num_steps,
                batch_size_gen=args.batch_size_gen,
                batch_size_ll=args.batch_size_ll,
                max_num_seq_per_generation=args.max_num_seq_per_generation,
                window_size=args.window_size,
                step_size=args.step_size,
                exclusions=exclusions,
                freeze_mutation_locations=args.freeze_mutation_locations,
                compare_type=args.compare_type,
                **gen_kwargs,
            )
            json_repr = {"orig_sequence": seq, "generations": []}
            for seq_set in generations:
                json_repr["generations"].append(
                    [x.as_json_dict() for x in seq_set]
                )
            whandle.write(json.dumps(json_repr))
            whandle.write("\n")


def add_sampler_parameters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gen_do_sample",
        help="Perform sampling instead of greedy",
        default=None,
        action="store_true",
    )

    parser.add_argument(
        "--gen_num_beams",
        help="If Beam Search is used, number of beams",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--gen_temperature",
        help="Temperature for generation",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--gen_top_k",
        help="k for top-k sampling algorithm",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--gen_top_p",
        help="Nucleus size for nucleus sampling",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--gen_max_new_tokens",
        help="Maximum number of new tokens to sample",
        default=96,
        type=int,
    )

    parser.add_argument(
        "--gen_num_beam_groups",
        help="Number of beam groups (positive value will disallow sampled beam search)",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--gen_diversity_penalty",
        help="Penalizes lack of diversity among beam groups (only for group beam search)",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--gen_num_return_sequences",
        help="Number of sequences to return",
        default=None,
        type=int,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict likely evolutionary path of a sequence")

    parser.add_argument(
        "--ref",
        help="File containing reference sequence",
        required=True,
    )

    parser.add_argument(
        "--seq",
        help="Fasta file containing sequences to mutate",
        required=True,
    )

    parser.add_argument(
        "--num_steps",
        help="Number of evolutionary steps to predict",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output_prefix",
        help="Prefix of output file",
        required=True,
    )

    parser.add_argument(
        "--model_checkpoint",
        help="Checkpoint of model to use",
        required=True,
    )

    parser.add_argument(
        "--cuda",
        help="Use CUDA GPU",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--batch_size_gen",
        help="Batch size for model generation",
        default=16,
        type=int,
    )

    parser.add_argument(
        "--batch_size_ll",
        help="Batch size for likelihood calculation",
        default=16,
        type=int,
    )

    parser.add_argument(
        "--max_num_seq_per_generation",
        help="Maximum number of top sequences to retain per generation",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--window_size",
        help="Window size of predictions",
        default=64,
        type=int,
    )

    parser.add_argument(
        "--step_size",
        help="Step size of predictions",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--exclusions",
        help="Comma-separated list of files indicating sequences to exclude from the generated cases",
        default=None,
    )

    parser.add_argument(
        "--freeze_mutation_locations",
        help="Freeze locations with mutations",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--compare_type",
        help="Metric based on which sequence superiority is determined",
        default="LL",
        choices=["LL", "LL_RATIO"],
    )

    add_sampler_parameters(parser)

    args = parser.parse_args()

    args.compare_type = CompareType.LL if args.compare_type == "LL" else CompareType.LL_RATIO

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=f"{args.output_prefix}.log",
    )

    main(args)
