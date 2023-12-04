# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pysam
from argparse import ArgumentParser
import random
import math
from utils import get_full_sequence
from pandas_utils import read_data_frame_cached
import datetime
from collections import defaultdict
from functools import reduce
from operator import concat
from typing import Union
from functools import lru_cache
from typing import Callable
import copy
import pandas
import logging

_MEANING_OF_EVERYTHING = 42

random.seed(_MEANING_OF_EVERYTHING)

logger = logging.getLogger(__file__)


@lru_cache(maxsize=1024)
def get_full_sequence_cached(*args, **kwargs):
    return get_full_sequence(*args, **kwargs)


def parse_date(x: str) -> datetime.datetime:
    return datetime.datetime.strptime(x, "%Y-%m-%d")


def concat_sequences(*args) -> list:
    return reduce(lambda a, b: concat(a, b), args, [])


def write_full_seq(seqs: Union[list, set], ref: str, output_name: str) -> None:
    ref_count = defaultdict(int)

    with open(output_name, "w") as fhandle:
        for seq in seqs:
            full_seq = get_full_sequence_cached(seq.strip(), ref)
            seq_name = "REF" if len(seq.strip()) == 0 else seq.replace(",", "_")
            seq_count = ref_count[seq_name]
            fhandle.write(f">{seq_name}_{seq_count}\n{full_seq}\n")
            ref_count[seq_name] += 1


def write_mutation_seq(seqs: Union[list, set], output_name: str) -> None:
    with open(output_name, "w") as fhandle:
        for seq in set(seqs):
            fhandle.write(seq + "\n")


def get_enumerated_list(x: Union[list, set], c: dict) -> list:
    results = []
    for i in x:
        results.extend([i] * c[i])
    return results


def split_by_lineage(
    df: pandas.DataFrame,
    n_train_per_bucket: int,
    n_val_per_bucket: int,
    n_test_per_bucket: int,
    randshuffler: Callable = random.shuffle,
):
    logger.info("Bucketized split of pango lineages")

    df = df.groupby(["PangoLineage"], as_index=False)["Accession ID"].count()

    pango_counts = sorted(zip(
        df.PangoLineage, df["Accession ID"]), key=lambda x: (x[1], x[0]), reverse=True)

    logger.info(f"Found {len(pango_counts)} buckets")

    block_size = n_train_per_bucket + n_val_per_bucket + n_test_per_bucket

    train, val, test = [], [], []

    for i in range(0, len(pango_counts), block_size):
        bucket = [x[0] for x in pango_counts[i: i + block_size]]
        randshuffler(bucket)
        train.extend(bucket[: n_train_per_bucket])
        val.extend(bucket[n_train_per_bucket: n_train_per_bucket + n_val_per_bucket])
        test.extend(bucket[n_train_per_bucket + n_val_per_bucket: ])

    logger.info(f"Found {len(train)}, {len(val)}, {len(test)} train, val, test respectively")

    return train, val, test


def main(args):
    with open(args.ref) as fhandle:
        ref = fhandle.read().strip()

    df = read_data_frame_cached(args.tsv, protein=args.protein, datefield=args.datefield)
    df = df[df.ParsedDate <= parse_date(args.last_date)]
    orig_df = df

    train_lineages, val_lineages, test_lineages = split_by_lineage(
        df,
        args.n_train_per_bucket,
        args.n_val_per_bucket,
        args.n_test_per_bucket,
        args.randshuffler,
    )

    df = df.groupby(["SpikeMutations", "PangoLineage"], as_index=False)["Accession ID"].count()

    spike_to_pango_dict = defaultdict(list)
    pango_to_spike_dict = defaultdict(list)
    spike_to_count_dict = defaultdict(int)

    for spike, pl, c in zip(df.SpikeMutations, df.PangoLineage, df["Accession ID"]):
        spike_to_pango_dict[spike].append(pl)
        pango_to_spike_dict[pl].append(spike)
        spike_to_count_dict[spike] += c

    train_sequences = set(concat_sequences(*[pango_to_spike_dict[t] for t in train_lineages]))
    val_sequences = set(concat_sequences(*[pango_to_spike_dict[v] for v in val_lineages]))
    test_sequences = set(concat_sequences(*[pango_to_spike_dict[t] for t in test_lineages]))

    move_to_train_list = train_sequences.intersection(val_sequences)
    val_sequences = val_sequences.difference(move_to_train_list)

    move_to_train_list = train_sequences.intersection(test_sequences)
    test_sequences = test_sequences.difference(move_to_train_list)

    if args.first_date:
        logger.info(f"Found first date {args.first_date}. Only keeping sequences in circulation after this date.")
        first_date = parse_date(args.first_date)
        seq_after_first_date = set(orig_df[orig_df.ParsedDate >= first_date].SpikeMutations.tolist())

        def filter_helper(item_list: list) -> list:
            return [x for x in item_list if x in seq_after_first_date]

        train_sequences, val_sequences, test_sequences = [filter_helper(x) for x in [
            train_sequences, val_sequences, test_sequences,
        ]]

    if args.enumerate:
        train_sequences = get_enumerated_list(train_sequences, spike_to_count_dict)
        val_sequences = get_enumerated_list(val_sequences, spike_to_count_dict)
        test_sequences = get_enumerated_list(test_sequences, spike_to_count_dict)
        logger.info(
            f"Found {len(train_lineages)} train and {len(val_lineages)} val lineages "
            f"{len(test_lineages)} test lineages after enumerating")

    write_full_seq(train_sequences, ref, args.prefix + ".train.fa")
    write_mutation_seq(train_sequences, args.prefix + ".train.mutations.lst")
    write_mutation_seq(train_lineages, args.prefix + ".train.lineages.lst")

    write_full_seq(val_sequences, ref, args.prefix + ".val.fa")
    write_mutation_seq(val_sequences, args.prefix + ".val.mutations.lst")
    write_mutation_seq(val_lineages, args.prefix + ".val.lineages.lst")

    if test_sequences:
        write_full_seq(test_sequences, ref, args.prefix + ".test.fa")
        write_mutation_seq(test_sequences, args.prefix + ".test.mutations.lst")
        write_mutation_seq(test_lineages, args.prefix + ".test.lineages.lst")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = ArgumentParser(description="Split a fasta into two sets")

    parser.add_argument("--prefix", help="Output prefix", required=True)
    parser.add_argument("--tsv", help="TSV variants file", required=True)
    parser.add_argument("--last_date", help="Last date of sequence occurrence", required=True)
    parser.add_argument("--first_date", help="First date of sequence occurrence", required=False)
    parser.add_argument("--ref", help="Reference sequence file", required=True)
    parser.add_argument("--n_train_per_bucket", help="Number of train per bucket", default=4, type=int)
    parser.add_argument("--n_val_per_bucket", help="Number of val per bucket", default=1, type=int)
    parser.add_argument("--n_test_per_bucket", help="Number of test per bucket", default=1, type=int)
    parser.add_argument("--protein", help="Protein for which analysis is done", default="Spike")
    parser.add_argument("--datefield", help="Date field", default="Submission date")
    parser.add_argument("--enumerate", help="Enumerate sequences based on occurrence count", default=False, action="store_true")

    args = parser.parse_args()

    args.randshuffler = random.shuffle

    main(args)
