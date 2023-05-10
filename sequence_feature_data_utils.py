# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import torch
import models
from data_processing import MemmapDataset, Tokenizer
from typing import Optional, Generator, Callable, Tuple, List
import os
import json
import numpy as np
import logging
import pickle
from collections import namedtuple, OrderedDict
from collections.abc import Iterable
import random
import datetime
import itertools
from utils import SpecialTokens, _DEFAULT_SPECIAL_TOKENS, get_full_sequence
import pandas
from collections import defaultdict
import tqdm
from create_occurrence_buckets import get_training_pairs_from_tsv, SeqPair
import math

logger = logging.getLogger(__file__)


class EmbeddingsMemmapData:
    """
    Create embeddings data and store on disk
    """
    def __init__(
        self,
        data_path: str,
        n_items: Optional[int] = None,
        max_length: Optional[int] = None,
        embedding_dim: Optional[int] = None,
    ):
        self.data_path = data_path

        if not n_items or not max_length or not embedding_dim:
            with open(self.config_file) as fhandle:
                config = json.load(fhandle)

            self.n_items = config["n_items"]
            self.max_length = config["max_length"]
            self.embedding_dim = config["embedding_dim"]

            with open(self.mappings_file, "rb") as fhandle:
                self.mappings = pickle.load(fhandle)

            mode = "r"
            logger.info("Opening files for reading")
        else:
            mode = "w+"
            os.makedirs(data_path)
            self.n_items = n_items
            self.max_length = max_length
            self.embedding_dim = embedding_dim
            logger.info("Opening files for writing")
            self.mappings = OrderedDict()

        self.embeddings = np.memmap(
            os.path.join(data_path, "embeddings.memmap"),
            shape=(self.n_items, self.max_length, self.embedding_dim),
            dtype=np.float32,
            mode=mode,
        )
        self.mask = np.memmap(
            os.path.join(data_path, "attention_mask.memmap"),
            shape=(self.n_items, self.max_length),
            dtype=np.uint8,
            mode=mode,
        )
        self.mode = mode

    @property
    def config_file(self):
        return os.path.join(self.data_path, "config.json")

    @property
    def mappings_file(self):
        return os.path.join(self.data_path, "mappings.pkl")

    def __setitem__(self, mutation_seq: str, item: tuple) -> None:
        if mutation_seq in self.mappings:
            idx = self.mappings[mutation_seq]
            logger.warning(f"Overwriting idx {idx} with mutations {mutation_seq}")
        else:
            idx = len(self.mappings)
            self.mappings[mutation_seq] = idx

        self.embeddings[idx] = item[0]
        self.mask[idx] = item[1]

    def __getitem__(self, mutation_seq: str) -> tuple:
        if mutation_seq in self.mappings:
            idx = self.mappings[mutation_seq]
            embeddings = np.array(self.embeddings[idx])
            mask = np.array(self.mask[idx])
            return torch.Tensor(embeddings), torch.ByteTensor(mask)

        raise ValueError(f"Mutation sequence {mutation_seq} not found")

    def __len__(self):
        return self.n_items

    def close(self):
        if self.mode == "r":
            raise AttributeError("Cannot call close in read only mode")

        self.embeddings.flush()
        self.mask.flush()

        with open(self.mappings_file, "wb") as fhandle:
            pickle.dump(self.mappings, fhandle)

        with open(self.config_file, "w") as fhandle:
            json.dump(
                {"n_items": self.n_items,
                "max_length": self.max_length,
                "embedding_dim": self.embedding_dim}, fhandle)


def get_weekly_discoveries(
    df: pandas.DataFrame,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    period: datetime.timedelta,
) -> Tuple[list, dict]:
    current_date = start_date
    next_date = current_date + period
    sequence_buckets = []
    all_sequences = dict()

    while next_date < end_date:
        df_slice = df[(df.ParsedDate >= current_date) & (df.ParsedDate < next_date)]
        sequences_in_period = df_slice.SpikeMutations.tolist()
        for seq in sequences_in_period:
            all_sequences[seq] = len(sequence_buckets)
        sequence_buckets.append(sequences_in_period)
        current_date = next_date
        next_date = next_date + period

    return sequence_buckets, all_sequences


def get_sequence_counts(df: pandas.DataFrame) -> dict:
    counts = defaultdict(int)

    for item in tqdm.tqdm(df.itertuples(), desc="Counting"):
        counts[item.SpikeMutations] += 1

    return {key: value for key, value in counts.items()}


def prepare_pariwise_data_preliminaries(
    df: pandas.DataFrame,
    discovery_end_date: datetime.datetime,
    availability_end_date: datetime.datetime,
    period_length: int = 7,
    min_date: Optional[datetime.datetime] = None,
    protein: str = "Spike",
):
    logger.info("Finding sequences discovered in each period, and sequence counts")

    if not min_date:
        min_date = df.ParsedDate.min()
    
    sequence_buckets, all_sequences = get_weekly_discoveries(
        df.loc[df.groupby(f"{protein}Mutations").ParsedDate.idxmin()],
        start_date=min_date,
        end_date=discovery_end_date,
        period=datetime.timedelta(days=period_length)
    )

    sequence_counts = get_sequence_counts(
        df[df.ParsedDate < availability_end_date]
    )

    return sequence_buckets, all_sequences, sequence_counts


class PairwiseDataset(torch.utils.data.Dataset):
    """
    Create a pairwise dataset where sequences are paired with each other
    and the labels represent a continuum
    """
    def __init__(
        self,
        sequence_buckets: list,
        all_sequences: dict,
        sequence_counts: dict,
        max_sample_steps: int = 128,
        embeddings: Optional[EmbeddingsMemmapData] = None,
        special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
        ref: Optional[str] = None,
        randint_functor: Callable = random.randint,
        randsample_functor: Callable = random.sample,
        randshuffle_functor: Callable = random.shuffle,
    ):
        super().__init__()

        self.sequence_buckets = sequence_buckets
        self.all_sequences = all_sequences
        self.sequence_counts = sequence_counts
        self.all_sequences_list = list(self.all_sequences.keys())
        self.tokenizer = Tokenizer()
        self.max_sample_steps = max_sample_steps
        self.embeddings = embeddings
        self.special_tokens = special_tokens
        self.ref = ref
        self.randint_functor = randint_functor
        self.randsample_functor = randsample_functor
        self.randshuffle_functor = randshuffle_functor

    def precompute_pairings(self, max_items_per_bucket: int = -1):
        logger.info("Precomputing pairings")

        self.pairings = []

        for bucket in self.sequence_buckets:
            if not all(b in self.all_sequences for b in bucket):
                continue

            combinations = list(itertools.permutations(bucket, 2))

            if max_items_per_bucket <= 0:
                sample_size = len(bucket)
            else:
                l = len(bucket)
                sample_size = min(max_items_per_bucket, len(combinations))

            for selection in self.randsample_functor(combinations, sample_size):
                self.pairings.append(selection)

    def __len__(self) -> int:
        if hasattr(self, "pairings"):
            return len(self.pairings)

        return len(self.all_sequences_list)

    def _get_seq(self, idx: int) -> tuple:
        if hasattr(self, "pairings"):
            seq_ordering = self.pairings[idx]
        else:
            seq = self.all_sequences_list[idx]
            seq_bucket_idx = self.all_sequences[seq]
            seq_bucket = self.sequence_buckets[seq_bucket_idx]
            paired_seq = None

            for i in range(self.max_sample_steps):
                paired_seq = self.randsample_functor(seq_bucket, 1)[0]
                if paired_seq != seq:
                    break
            else:
                return None

            flag = self.randint_functor(0, 1)

            if flag == 0:
                seq_ordering = (seq, paired_seq)
            else:
                seq_ordering = (paired_seq, seq)

        frac = self.sequence_counts[seq_ordering[1]] / (
            self.sequence_counts[seq_ordering[0]] + self.sequence_counts[seq_ordering[1]])

        result = (seq_ordering, frac)

        return result

    def _tokenizer_helper(self, seq: str) -> list:
        full_seq = get_full_sequence(seq, self.ref)
        seq_to_tokenize = [self.special_tokens.start_of_sequence] + list(full_seq) + [self.special_tokens.end_of_sequence]
        return [self.tokenizer.mapper[i] for i in seq_to_tokenize]

    def __getitem__(self, idx):
        res = self._get_seq(idx)

        if res is None:
            return None
        else:
            seq_ordering, frac = res

        if self.embeddings:
            seq0_embeddings = self.embeddings[seq_ordering[0]]
            seq1_embeddings = self.embeddings[seq_ordering[1]]
            return (seq0_embeddings, seq1_embeddings), frac
        else:
            tokenized0 = torch.LongTensor(self._tokenizer_helper(seq_ordering[0]))
            tokenized1 = torch.LongTensor(self._tokenizer_helper(seq_ordering[1]))
            return (tokenized0, tokenized1), frac


def collate_embeddings(batch: list) -> tuple:
    def collate_helper(batch: list) -> tuple:
        embeddings, masks = tuple(zip(*batch))
        return torch.stack(embeddings, dim=0), torch.stack(masks, dim=0)

    embeddings, labels = tuple(zip(*batch))
    seq0_embeddings, seq1_embeddings = tuple(zip(*embeddings))

    return (
        collate_helper(seq0_embeddings),
        collate_helper(seq1_embeddings),
        torch.Tensor(labels)
    )


def collate_tokens(batch: list) -> tuple:
    def collate_helper(batch: list) -> tuple:
        max_length = max(x.shape[0] for x in batch)
        input_ids = torch.zeros(len(batch), max_length).long()
        attention_mask = torch.zeros(len(batch), max_length).byte()
        for i, b in enumerate(batch):
            input_ids[i, :b.shape[0]] = b
            attention_mask[i, :b.shape[0]] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenized_sequences, labels = tuple(zip(*batch))
    seq0_tokens, seq1_tokens = tuple(zip(*tokenized_sequences))

    return collate_helper(seq0_tokens), collate_helper(seq1_tokens), torch.Tensor(labels)


def collate_function(batch: list):
    batch = [b for b in batch if b]
    first_item, first_label = batch[0]

    if type(first_item[0]) is tuple:
        return collate_embeddings(batch)
    else:
        return collate_tokens(batch)


def tokenize_helper(seq: str, ref: str, special_tokens: SpecialTokens, mapper: dict) -> list:
    full_seq = get_full_sequence(seq, ref)
    eos_token = mapper[special_tokens.end_of_sequence]

    assert(special_tokens.start_of_sequence not in full_seq), "Start of sequence not expected in sequence"
    assert(mapper[full_seq[-1]] == eos_token), "Expected sequence to end in end of sequence token"

    seq_to_tokenize = [special_tokens.start_of_sequence] + list(full_seq)

    return [mapper[x] for x in seq_to_tokenize]


class PairwiseBucketizedDataset(torch.utils.data.Dataset):
    """
    Create a simple pairwise dataset when sequence pairs are already given
    """
    def __init__(
        self,
        sequence_pairings: List[SeqPair],
        ref: str,
        special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
        randint_functor: Callable = random.randint,
        min_max_occurrence: Optional[float] = None,
        min_min_occurrence: Optional[float] = None,
        n_leading_week_counts: Optional[int] = None,
    ):
        super().__init__()

        self.pairings = []

        for x in sequence_pairings:
            if min_max_occurrence and max(x.count0, x.count1) < min_max_occurrence:
                continue
            if min_min_occurrence and min(x.count0, x.count1) < min_min_occurrence:
                continue
            self.pairings.append(x)

        self.n_leading_week_counts = n_leading_week_counts
        self.all_entries_have_week_counts = all(
            x.weekly_counts0 and x.weekly_counts1 for x in self.pairings
        )

        if n_leading_week_counts and not self.all_entries_have_week_counts:
            raise AttributeError(
                "Cannot use leading week counts when all entries do not have week counts")

        self.ref = ref
        self.special_tokens = special_tokens
        self.tokenizer = Tokenizer()
        self.randint_functor = randint_functor

    def __len__(self):
        return len(self.pairings)

    def _tokenizer_helper(self, seq: str) -> list:
        return tokenize_helper(seq, self.ref, self.special_tokens, self.tokenizer.mapper)

    def get_theoretical_min_loss(self):
        total_entropy = 0

        for p in self.pairings:
            total = p.count0 + p.count1
            p0 = p.count0 / total
            p1 = p.count1 / total
            total_entropy += p0 * math.log(p0 + 1e-12) + p1 * math.log(p1 + 1e-12)

        return -total_entropy / len(self.pairings)

    def __getitem__(self, idx: int) -> tuple:
        seq_pair = self.pairings[idx]

        seq0 = torch.LongTensor(self._tokenizer_helper(seq_pair.seq0))
        seq1 = torch.LongTensor(self._tokenizer_helper(seq_pair.seq1))

        if self.n_leading_week_counts:
            n_leading = self.n_leading_week_counts
            leading_weeks0 = seq_pair.weekly_counts0[: n_leading]
            leading_weeks1 = seq_pair.weekly_counts1[: n_leading]
        else:
            leading_weeks0 = None
            leading_weeks1 = None

        coin = self.randint_functor(0, 1)

        returns = []

        if coin == 1:
            frac = seq_pair.count1 / (seq_pair.count0 + seq_pair.count1)
            returns = [(seq0, seq1), (leading_weeks0, leading_weeks1), frac]
        else:
            frac = seq_pair.count0 / (seq_pair.count0 + seq_pair.count1)
            returns = [(seq1, seq0), (leading_weeks1, leading_weeks0), frac]

        if not self.n_leading_week_counts:
            returns = [returns[0], returns[-1]]

        return returns
