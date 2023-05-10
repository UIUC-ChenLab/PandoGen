# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import models
import competition_models
import torch
from typing import List, Optional, Callable, Union, Generator
from transformers import Trainer, BertLMHeadModel, TrainingArguments
from collections import OrderedDict
from functools import partial
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import EvalLoopOutput
import random
import logging
import tqdm
from utils import _DEFAULT_SPECIAL_TOKENS, SpecialTokens
from data_processing import Tokenizer
import math
import pysam
import tqdm
import torch.distributed
import json
from prediction_tools import calc_likelihoods

logger = logging.getLogger(__file__)


def is_sequence_valid(sequences: torch.LongTensor, eos_token: int, dim: int = 1) -> torch.BoolTensor:
    return torch.any(sequences == eos_token, dim=dim)


def pad_equally(seq_a: torch.Tensor, seq_b: torch.Tensor) -> tuple:
    max_len = max(seq_a.shape[1], seq_b.shape[1])
    pad_a = max_len - seq_a.shape[1]
    pad_b = max_len - seq_b.shape[1]

    if pad_a > 0:
        seq_a = torch.nn.functional.pad(seq_a, (0, pad_a, 0, 0))

    if pad_b > 0:
        seq_b = torch.nn.functional.pad(seq_b, (0, pad_b, 0, 0))

    return seq_a, seq_b


def membership_test(seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.BoolTensor:
    b_a = seq_a.shape[0]
    b_b = seq_b.shape[0]

    seq_a = seq_a[:, None, :].repeat(1, b_b, 1)  # (b_a, b_b, L)
    seq_b = seq_b[None].repeat(b_a, 1, 1)  # (b_a, b_b, L)

    return torch.any(torch.all(seq_a == seq_b, dim=2), dim=0)  # [b_b]


def membership_test_batched(seq_a: torch.Tensor, seq_b: torch.Tensor, batch_size: int = 2 ** 16) -> torch.BoolTensor:
    if seq_a.shape[0] <= batch_size:
        return membership_test(seq_a, seq_b)

    chunks = torch.split(seq_a, split_size_or_sections=batch_size, dim=0)
    results = None

    for c in chunks:
        r = membership_test(c, seq_b)
        if results is None:
            results = r
        else:
            results = torch.logical_or(results, r)

    return results


def get_attn_mask(sequences: torch.LongTensor, eos_token: int) -> torch.ByteTensor:
    mask = torch.nn.functional.pad(
        torch.cumprod(torch.ne(sequences, eos_token), dim=1),
        pad=[1, 0, 0, 0], value=1)[:, :-1]
    return mask


class RewardModel(torch.nn.Module):
    def __init__(
        self,
        scorer: torch.nn.Module,
        eos_token: int,
        prev_sequences: torch.ByteTensor,
    ):
        super().__init__()
        self.scorer = scorer
        self.eos_token = eos_token
        self.register_buffer("prev_sequences", prev_sequences)

    def _get_existing_flag_decoder(
        self,
        sequences: torch.LongTensor,
    ) -> torch.BoolTensor:
        # For membership, we ignore the bos token, then pad equally
        sequences = sequences[:, 1:]
        prev_sequences, sequences = pad_equally(self.prev_sequences, sequences)
        membership = membership_test(prev_sequences.byte(), sequences.byte())
        return membership

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.ByteTensor) -> torch.LongTensor:
        input_ids = input_ids.masked_fill(attention_mask == 0, 0)
        potentials = self.scorer({"input_ids": input_ids, "attention_mask": attention_mask})
        membership = self._get_existing_flag_decoder(input_ids)
        sequence_valid = is_sequence_valid(input_ids, self.eos_token)
        return potentials, membership, sequence_valid


def get_quantile(quantile_map: list, p: float) -> int:
    for i, q_map in enumerate(quantile_map):
        if q_map[0] <= p < q_map[1]:
            return i

    raise ValueError(f"Potential {p} has no place in {quantile_map}")


def assign_quantiles(
    sequences: dict,
    potentials: torch.Tensor,
    membership: torch.Tensor,
    sequence_valid: torch.Tensor,
    quantiles: list,
) -> torch.LongTensor:
    """
    Assign quantiles to the given sequences. Accepts, sequences, quantile specifications and
    output of the reward model as inputs.
    """
    potentials = potentials[:, 0].cpu().tolist()
    membership = membership.cpu().tolist()
    sequence_valid = sequence_valid.cpu().tolist()
    quantile_assignments = []

    for i, (p, m, v) in enumerate(zip(potentials, membership, sequence_valid)):
        input_ids = sequences["input_ids"][i]
        attention_mask = sequences["attention_mask"][i]

        if m:
            quantile_assignments.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "quantile": len(quantiles)
                }
            )
            continue

        q = get_quantile(quantiles, p)

        quantile_assignments.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "quantile": q
            }            
        )

    return quantile_assignments


def generation_summary(
    potentials: torch.Tensor,
    membership: torch.BoolTensor,
    sequence_valid: torch.BoolTensor
) -> tuple:
    """
    Obtain a summary of the quality of the generations as evaluated by the reward model
    """
    all_valid_potentials = []
    all_valid_novel_potentials = []
    num_valid = 0
    num_valid_novel = 0
    total = 0

    for p, m, v in zip(
        potentials.cpu().tolist(), membership.cpu().tolist(), sequence_valid.cpu().tolist()):

        if v:
            all_valid_potentials.append(p)
            if not m:
                all_valid_novel_potentials.append(p)

        num_valid += int(v)
        num_valid_novel += int(v and not m)
        total += 1

    return [
        sum(all_valid_novel_potentials) / total,
        sum(all_valid_potentials) / total,
        num_valid_novel,
        num_valid,
    ]


def calc_model_loss_quark_sequence(
    train_sequences: dict, logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    logits [1: -1] should be used. logits[1] is the result of sending in the second token,
    that is, the quantile token in. We exclude the last one because, there is no token
    generated based on the last position.

    input_ids[2] onwards include the actual sequence tokens. Hence we calculate the log-prob
    for those tokens. We also use attention_mask[2:] to do the masking as it correctly aligns
    with the input_ids
    """
    mask_3d = train_sequences["attention_mask"][:, 2:, None]
    dist = torch.distributions.categorical.Categorical(logits=logits[:, 1:-1])
    ll = dist.log_prob(train_sequences["input_ids"][:, 2:]).masked_fill(
        mask_3d[:, :, 0] == 0, 0)
    num_tokens = torch.clamp(torch.sum(mask_3d).float(), min=eps)
    return -torch.sum(ll) / num_tokens, mask_3d


def quark_forward(self, train_sequences: dict, ref_sequences: dict) -> tuple:
    """
    Quark forward function for testing purposes

    ref_logits are valid from 0 to -1. ref_logits[0] is based on inputting the bos
    token. The last value doesn't matter since nothing is generated based on that.

    model_logits are valid from 1 to -1, since model_logits contain an additional
    quantile bos token at the start. Hence for KL-divergence these two sets of tensors
    are to be aligned.

    Regarding calculation of likelihoods please refer to calc_model_loss_quark_sequence above.
    """
    with torch.no_grad():
        ref_logits = self.ref_model(**ref_sequences).logits
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)[:, : -1]
        ref_probs = torch.exp(ref_log_probs)

    model_results = self.train_model(**train_sequences)
    model_loss, mask_3d = calc_model_loss_quark_sequence(train_sequences, model_results.logits)
    model_log_probs = torch.log_softmax(model_results.logits, dim=-1)[:, 1: -1]

    masked_pointwise_kldiv = (ref_probs * (ref_log_probs - model_log_probs)).masked_fill(
        mask_3d == 0, 0)

    n_tokens = torch.sum(mask_3d)

    return model_loss, torch.sum(masked_pointwise_kldiv) / n_tokens


class QuarkModel(torch.nn.Module):
    def __init__(
        self,
        train_model: BertLMHeadModel,
        ref_model: BertLMHeadModel,
        reward_model: RewardModel,
        bos_token_ref: int,
    ):
        """
        A Quark model pair with train, reward, and ref models. Note that the ref model and
        the reward model are always in eval state. We do the wrapping here because it allows
        the models to be moved to the GPU together, and provides a better API
        """
        super().__init__()
        self.train_model = train_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.bos_token_ref = bos_token_ref
        self.ref_model.eval()
        self.reward_model.eval()

    def eval(self) -> torch.nn.Module:
        """
        Sets Quark's training model in eval mode
        """
        self.train_model.eval()
        return self

    def train(self, mode: bool = True) -> torch.nn.Module:
        """
        Sets Quark's train_model in training mode
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.train_model.train(mode)
        return self

    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except StopIteration:
            return False

    def get_device(self):
        return next(self.parameters()).get_device()

    def generate(
        self,
        bos_token_id: Union[int, list],
        eos_token_id: int,
        gen_kwargs: dict,
    ) -> torch.LongTensor:
        """
        This generates sequences from the training model
        """
        if isinstance(bos_token_id, list):
            bos = torch.LongTensor(bos_token_id)[None]

            if self.is_cuda:
                bos = bos.to(self.get_device())

            # If bos is a sequence, set that as input_ids
            res = self.train_model.generate(
                input_ids=bos,
                eos_token_id=eos_token_id,
                **gen_kwargs,
            )
            # Splice out the bos section, except for the "[CLS]" token
            return torch.cat((res[:, :1], res[:, len(bos_token_id):]), dim=1)
        else:
            # If bos is a token, simply set it as a bos token
            return self.train_model.generate(
                bos_token_id=bos_token_id, eos_token_id=eos_token_id, **gen_kwargs)

    def get_rewards(self, input_ids: torch.LongTensor, attention_mask: torch.ByteTensor, *args, **kwargs) -> tuple:
        """
        Run reward model with no gradient accumulation
        """
        with torch.no_grad():
            """
            The reward model expects the BOS token to be the reference case "[CLS]". The
            following assignment is redundant as in the new generation code the quantile
            token is stripped away.
            """
            input_ids_for_rewards = input_ids.clone()
            input_ids_for_rewards[:, 0] = self.bos_token_ref
            res = self.reward_model(input_ids, attention_mask)

        return res

    def forward(self, train_sequences: dict, ref_sequences: dict, *args, **kwargs) -> tuple:
        """
        This runs the quark finetuning step
        """
        return quark_forward(self, train_sequences, ref_sequences)


def calc_likelihoods_top(
    model: torch.nn.Module,
    res: torch.LongTensor,
    is_quark_model: bool = False,
    eos_token: Optional[int] = None,
    temperature: Optional[float] = None,
) -> torch.Tensor:
    logits = model(input_ids=res).logits
    offset = 2 if is_quark_model else 1

    seq = res[:, offset:]
    seq_logits = logits[:, offset - 1: -1]

    if temperature is not None:
        seq_logits = seq_logits / temperature

    if eos_token is None:
        mapper = Tokenizer().mapper
        eos_token = mapper[_DEFAULT_SPECIAL_TOKENS.end_of_sequence]

    likelihoods = calc_likelihoods(
        seq,
        end_token_value=eos_token,
        logits=seq_logits,
    )
    return likelihoods


class LikelihoodScorer(torch.nn.Module):
    """
    Simply use a decoder model's likelihood score as reward
    """
    def __init__(self, decoder: BertLMHeadModel):
        super().__init__()
        self.decoder = decoder

    def forward(self, sequence_dict: dict, *args, **kwargs):
        return calc_likelihoods_top(
            self.decoder, sequence_dict["input_ids"], is_quark_model=False)[:, None]


def concat_batches(batches: List[torch.Tensor]) -> torch.Tensor:
    max_length = max(b.shape[1] for b in batches)
    all_batches = torch.cat(
        [torch.nn.functional.pad(b, pad=(0, max_length - b.shape[1])) for b in batches], dim=0)
    return all_batches


def determine_quantiles(scores: List[float], quantile_spec: List[float]) -> list:
    """
    Determine quantile cutoffs following the quantile_spec. The quantile spec is a list
    of floating point values indicating the end-point of the quantiles as fractions. The nth
    fraction in the quantile spec indicates the fraction of sequences contained in quantiles
    1 to n (inclusive, and 1-based numbering). Quantiles as listed here are reversed
    for token ordering purposes.
    """
    sorted_scores = sorted(scores, reverse=True)
    n_sequences = len(scores)
    quantile_ranges = []

    logger.info(f"Score range = {sorted_scores[-1]} -> {sorted_scores[0]}")

    hi_score = float("inf")

    for q in quantile_spec:
        nth_seq = math.ceil(q * n_sequences)
        nth_score = sorted_scores[nth_seq - 1]
        score_range = (nth_score, hi_score)
        quantile_ranges.append(score_range)
        hi_score = nth_score

    quantile_ranges.append((-float("inf"), hi_score))

    return list(reversed(quantile_ranges))


def generate_datapool_batch(
    model: QuarkModel,
    gen_kwargs: dict,
    eos_token: int,
    quantile_offset: int,
    quantiles: Optional[list] = None,
    quantile_spec: Optional[List[float]] = None,
    bos_token: Optional[int] = None,
    n_batches: Optional[int] = None,
    return_quantiles: bool = False,
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
    batch_generator: Optional[Generator[Union[int, torch.Tensor], None, None]] = None,
) -> list:
    """
    Generate a batch of data and assign quantiles. We only generate from the highest quantile.

    :param model: The combined QuarkModel that does generation, reward evaluation, and forward
    :param gen_kwargs: Generation arguments
    :param eos_token: End-of-sequence token
    :param quantile_offset: The number of "regular" tokens to use after which quantile tokens begin
    :param quantiles: List of quantile ranges, [lo, hi)
    :param quantile_spec: Quantile specification for initialization. See determine_quantiles
    :param bos_token: The beginning of sentence token
    :param n_batches: Number of batches to generate
    :param return_quantiles: Whether quantiles should be returned
    :param special_tokens: Special token specification
    :param batch_generator: Generator of a single batch for the case where we have a pre-computed list
    """
    mapper = Tokenizer().mapper
    default_bos_token = mapper[special_tokens.start_of_sequence]

    if bos_token is None:
        if quantiles is None:
            raise ValueError("Both bos_token and quantiles cannot be None")
        bos_token = [default_bos_token, quantile_offset + len(quantiles) - 1]

    if quantiles is None:
        if quantile_spec is None:
            raise ValueError("Both quantiles and quantile_spec cannot be None")

    n_batches = 1 if n_batches is None else n_batches

    all_input_ids = []
    all_attention_mask = []
    all_potentials = []
    all_membership = []
    all_sequence_valid = []

    for i in tqdm.tqdm(range(n_batches), desc="Generating batches"):
        if batch_generator is None:
            batch = model.generate(
                bos_token_id=bos_token,
                eos_token_id=eos_token,
                gen_kwargs=gen_kwargs)
        else:
            batch = next(batch_generator)

        attention_mask = get_attn_mask(batch, eos_token)
        input_ids = batch.masked_fill(attention_mask == 0, 0)
        potentials, membership, sequence_valid = model.get_rewards(input_ids, attention_mask)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_potentials.append(potentials)
        all_membership.append(membership)
        all_sequence_valid.append(sequence_valid)

    input_ids = concat_batches(all_input_ids)
    attention_mask = concat_batches(all_attention_mask)
    potentials, membership, sequence_valid = [
        torch.cat(x, dim=0) for x in [all_potentials, all_membership, all_sequence_valid]]

    potentials = potentials.cpu()
    membership = membership.cpu()
    sequence_valid = sequence_valid.cpu()
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()

    if quantiles is None:
        quantiles = determine_quantiles(potentials[:, 0].cpu().tolist(), quantile_spec)

    results = assign_quantiles(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        potentials=potentials,
        membership=membership,
        sequence_valid=sequence_valid,
        quantiles=quantiles,
    )

    fixed_results = []

    for i, item in enumerate(results):
        new_item = dict()
        new_item["input_ids"] = item["input_ids"]
        new_item["attention_mask"] = item["attention_mask"]
        new_item["quantile_token"] = item["quantile"] + quantile_offset
        fixed_results.append(new_item)

    if return_quantiles:
        return fixed_results, (potentials, membership, sequence_valid), quantiles
    else:
        return fixed_results, (potentials, membership, sequence_valid)


def adjust_tensor_length(tensor_dict: dict, max_length: int, eos_token: int) -> dict:
    """
    This is length adjustment as needed for the data pool. A valid sequence (one with eos_token)
    may be come invalid due to truncation. Hence the valid signal is set here.
    """
    new_results = {}

    for key in tensor_dict:
        val = tensor_dict[key]

        if key in ["input_ids", "attention_mask"]:

            if val.shape[0] < max_length:
                pad_length = max_length - val.shape[0]
                val = torch.nn.functional.pad(val, pad=(0, pad_length))
            elif val.shape[0] > max_length:
                val = val[:max_length]

            if key == "input_ids":
                valid = is_sequence_valid(val, eos_token=eos_token, dim=0).item()

        new_results[key] = val

    new_results["valid"] = valid

    return new_results


def stratified_sampling(pool: list, sample_size: int):
    """
    Sample in a stratified way from each quantile. Sampling is done without replacement.

    A quantile can be uniformly selected as long as it has a valid member in it
    """

    # Split pool into quantile buckets
    pool_dict = {}
    for i, p in enumerate(pool):
        q = p["quantile_token"]
        if q not in pool_dict:
            pool_dict[q] = []
        pool_dict[q].append(i)

    # Shuffle each bucket so that ordered access is still random
    for key in pool_dict:
        random.shuffle(pool_dict[key])

    num_assigned_per_pool = {key: 0 for key in pool_dict}

    # Select a quantile at random, and take an item from it to
    # and put it in sampled pool. If the quantile bucket is empty,
    # remove the bucket from consideration going forward.
    sampled_pool = []

    for i in range(sample_size):
        # If only one quantile is left, and if that quantile is
        # already the max quantile, we stop adding more items into the pool
        if len(pool_dict) == 1:
            key = list(pool_dict.keys()).pop()
            if max(num_assigned_per_pool.values()) <= num_assigned_per_pool[key]:
                break
        
        # If no quantile is left, we against stop
        if not pool_dict:
            break

        quantile = random.sample(list(pool_dict.keys()), 1)[0]
        num_assigned_per_pool[quantile] += 1
        value = pool_dict[quantile].pop()
        sampled_pool.append(value)

        if len(pool_dict[quantile]) == 0:
            del pool_dict[quantile]

    logger.info(f"Pool assignments = {num_assigned_per_pool}")

    return sampled_pool


class DataPool(torch.utils.data.Dataset):
    def __init__(
        self,
        pool: list,
        eos_token: int,
        fixed_epoch_length: int = True,
        max_length: int = 1400,
        pool_length: Optional[int] = None,
        randsampler: Callable = random.sample,
        use_stratified_sample: bool = False,
    ):
        super().__init__()
        self.pool = []
        self.max_length = max_length
        self.eos_token = eos_token
        self.extend(pool)
        self._fixed_epoch_length = fixed_epoch_length
        self._pool_length = len(self.pool) if not pool_length else pool_length
        self.randsampler = randsampler
        self.subset_indices = list(range(len(self.pool)))
        self.use_stratified_sample = use_stratified_sample

    def __len__(self) -> int:
        if self._fixed_epoch_length:
            return min(self._pool_length, len(self.pool), len(self.subset_indices))
        else:
            return len(self.pool)

    def extend(self, pool: list):
        mapper = partial(
            adjust_tensor_length, max_length=self.max_length, eos_token=self.eos_token)
        self.pool.extend(filter(lambda x: x["valid"], map(mapper, pool)))

    def sample_epoch(self):
        if self.use_stratified_sample:
            self.subset_indices = stratified_sampling(self.pool, self._pool_length)
        else:
            self.subset_indices = self.randsampler(list(range(len(self.pool))), self._pool_length)

    def __getitem__(self, idx_: int) -> dict:
        idx = self.subset_indices[idx_]

        item = self.pool[idx]

        seq_input_ids = torch.nn.functional.pad(item["input_ids"], pad=(1, 0))
        seq_input_ids[0] = item["input_ids"][0]
        seq_input_ids[1] = item["quantile_token"]
        seq_attention_mask = torch.nn.functional.pad(item["attention_mask"], pad=(1, 0), value=1)

        ref_input_ids = item["input_ids"]
        ref_attention_mask = item["attention_mask"]

        return {
            "input_ids": seq_input_ids,
            "attention_mask": seq_attention_mask,
        }, {
            "input_ids": ref_input_ids,
            "attention_mask": ref_attention_mask,
        }


def collate_function(batch: list) -> dict:
    """
    Note that BertLMHead model auto-shifts labels to make sure predictions
    match the inputs. Code reference:
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/bert/modeling_bert.py#L1246
    """
    train_sequences, ref_sequences = tuple(zip(*batch))

    def helper(batch_part: list) -> dict:
        return {
            key: torch.stack([x[key] for x in batch_part], dim=0) for key in batch_part[0].keys()
        }

    train_sequences = helper(train_sequences)
    ref_sequences = helper(ref_sequences)

    return train_sequences, ref_sequences


def all_gather_list_torchrun(
    args: TrainingArguments,
    list_to_send: List[object],
    block_name: str = "[EVAL]",
) -> List[object]:
    if args.world_size > 1:
        if args.process_index == 0:
            logger.info(f"{block_name} Obtaining data from all processes")

        all_gathered = [None for i in range(args.world_size)]
        torch.distributed.all_gather_object(all_gathered, list_to_send)

        combined = []
        
        for g in all_gathered:
            combined.extend(g)

        return combined
    else:
        return list_to_send


def evaluation_loop(
    self,
    dataloader: torch.utils.data.DataLoader,
    description: str,
    prediction_loss_only: Optional[bool] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval",
) -> EvalLoopOutput:
    logger.info("""Running evaluation loop""")

    """
    Set model in evaluation
    """
    model = self.model.eval()

    """
    Sample the number of batches needed
    """
    generated_batches = []
    generated_rewards = []

    logger.info(f"Generating {self.args.n_eval_steps} batches of data")

    generated_batches, generated_rewards = generate_datapool_batch(
        model,
        self.args.gen_kwargs,
        self.args.eos_token,
        self.args.quantile_offset,
        self.args.quantiles,
        n_batches=self.args.n_eval_steps,
    )

    """
    Note: DataLoader in Trainer uses samplers to create batches. For both single-process
    case DataLoader (https://pytorch.org/docs/1.10/_modules/torch/utils/data/dataloader.html#DataLoader)
    recreates the iterator from scratch at the start of an epoch though the dataloader is the same object
    (Line https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L1339
    represents a call to the __iter__ function in the dataloader). For multiple data workers
    case, it resets the iterator.

    Hence, just adding to train_dataset automatically updates the train_dataloader to iterate
    through a different set of examples in the next epoch.
    
    For torchrun cases, this poses a problem as (1) there are multiple dataloader objects,
    one per each process, and (2) DistributedSampler is used in that case, and this sampler
    needs to know the datasets created from every process. We use torch.all_gather_object method
    to collect examples across every process. In addition, we replace sample_epoch with a
    custom sampling method in process 0.

    Note that DistributedSampler works off of the training dataset's length variable
    https://pytorch.org/docs/1.10/_modules/torch/utils/data/distributed.html#DistributedSampler
    So whether or not the DataLoader has copied the dataset, if the dataset length doesn't change,
    as is our use-case, everything should work well swimmingly.
    """
    generated_batches = all_gather_list_torchrun(self.args, generated_batches, "[EVAL]")
    self.train_dataset.extend(generated_batches)
    self.train_dataset.sample_epoch()

    if self.args.world_size > 1:
        torch.distributed.broadcast_object_list(self.train_dataset.subset_indices, src=0)

    """ Evaluate metrics """
    potentials, membership, sequence_valid = generated_rewards
    potentials = torch.squeeze(potentials, dim=1)
    generation_metrics = generation_summary(potentials, membership, sequence_valid)
    metrics = {"eval_loss": generation_metrics}

    return EvalLoopOutput(
        predictions=torch.stack([potentials, membership, sequence_valid], dim=-1),
        label_ids=None,
        metrics=metrics,
        num_samples=self.args.n_eval_steps,
    )


class QuarkTrainer(Trainer):
    def compute_loss(
        self,
        model: QuarkModel,
        inputs: dict,
        return_outputs: bool = False,
    ) -> Union[tuple, torch.Tensor]:
        model_loss, kl_div = model(*inputs)
        loss = (
            self.args.quark_alpha * model_loss + self.args.quark_beta * kl_div
        ) / self.args.quark_scale
        if return_outputs:
            return loss, None
        else:
            return loss

    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        eval_output = evaluation_loop(
            self, 
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if any(isinstance(cb, EarlyStoppingCallback) for cb in self.callback_handler.callbacks):
            loss = eval_output.metrics["eval_loss"]
            logger.info(f"Full loss vector={loss}, using {loss[0]} for early_stopping")
            eval_output.metrics["eval_loss"] = loss[0]

        return eval_output


def tensorize_prior_sequences(prior_sequences: str, max_length: int = 1400) -> torch.LongTensor:
    if prior_sequences is None:
        return torch.zeros(1, max_length).long()

    mapper = Tokenizer().mapper

    with pysam.FastaFile(prior_sequences) as fhandle:
        tokenized_sequences = [
            [mapper[i] for i in fhandle.fetch(r)] for r in fhandle.references]

    max_length = max(len(x) for x in tokenized_sequences)
    tensorized = torch.zeros(len(tokenized_sequences), max_length).long()

    for i, t in enumerate(tokenized_sequences):
        tensorized[i, : len(t)] = torch.LongTensor(t)

    return tensorized


def convert_pregen_to_pool(
    pregen_file: str,
    quark_model: QuarkModel,
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
    max_length: int = 1400,
    batch_size: int = 32,
    training_args: Optional[TrainingArguments] = None,
):
    """
    Drop-in replacement for model.generate in generate_datapool_batch
    """
    mapper = Tokenizer().mapper

    sequences = []

    with open(pregen_file, "r") as fhandle:
        for i, l in enumerate(fhandle):
            if (i + training_args.process_index) % training_args.world_size == 0:
                seq = json.loads(l)["seq"]
                if "[" in seq or "]" in seq:
                    continue
                sequences.append(seq)

    max_length_of_gen = max(len(s) for s in sequences) + 1  # Budget for bos token

    input_ids = torch.zeros(len(sequences), max_length_of_gen, dtype=torch.int64)

    for i, seq in enumerate(sequences):
        input_ids_single = torch.LongTensor(
            [mapper[special_tokens.start_of_sequence]] + [mapper[i] for i in seq])
        input_ids[i, :input_ids_single.shape[0]] = input_ids_single

    input_ids = input_ids[:, :max_length]

    # First yield the number of batches
    yield math.ceil(input_ids.shape[0] / batch_size)

    # Next, simply yield the list that has been split
    for i in torch.split(input_ids, split_size_or_sections=batch_size, dim=0):
        if quark_model.is_cuda:
            yield i.to(quark_model.get_device())
        else:
            yield i


def init_training(
    model: QuarkModel,
    quantile_spec: List[float],
    n_batches_init: int,
    gen_kwargs: dict,
    pool_size: Optional[int] = None,
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
    max_length: int = 1400,
    args: TrainingArguments = None,
    pregen: Optional[str] = None,
    use_stratified_sample: bool = False,
) -> tuple:
    """
    Initialize training by generating and scoring an initial batch
    """
    mapper = Tokenizer().mapper

    if torch.cuda.is_available():
        model.cuda()

    if pregen:
        batch_generator = convert_pregen_to_pool(
            pregen_file=pregen,
            quark_model=model,
            max_length=max_length,
            batch_size=gen_kwargs["num_return_sequences"],
            training_args=args,
        )
        n_batches_init = next(batch_generator)
    else:
        batch_generator = None

    pool, (potentials, memberhip, sequence_valid), quantiles = generate_datapool_batch(
        model,
        gen_kwargs=gen_kwargs,
        eos_token=mapper[special_tokens.end_of_sequence],
        quantile_offset=len(mapper),
        quantiles=None,
        quantile_spec=quantile_spec,
        bos_token=mapper[special_tokens.start_of_sequence],
        n_batches=n_batches_init,
        return_quantiles=True,
        batch_generator=batch_generator,
    )

    pool = all_gather_list_torchrun(args, pool, block_name="[INIT]")

    dataset = DataPool(
        pool,
        eos_token=mapper[special_tokens.end_of_sequence],
        fixed_epoch_length=True,
        max_length=max_length,
        pool_length=pool_size,
        use_stratified_sample=use_stratified_sample,
    )

    return dataset, quantiles
