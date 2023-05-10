# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import random
from typing import Callable, Generator, Optional, Union, List
from utils import SpecialTokens, get_full_sequence
import data_processing
import torch
import logging
import pandas
import datetime
from pandas_utils import read_data_frame_cached
from collections import defaultdict
from functools import reduce, partial
from operator import concat, itemgetter
import argparse
import tqdm
import os
import re
import math
from utils import mutation_positions_in_seq

logger = logging.getLogger(__file__)


def create_examples_for_sequence(
    sequence: str,
    min_masked_segment: int = 0,
    max_masked_segment: int = 128,
    randint_functor: Callable = random.randint,
    special_tokens: SpecialTokens = data_processing._DEFAULT_SPECIAL_TOKENS
) -> list:
    """
    Create examples covering each segment in a sequence
    """
    ptr = 0

    prev_masked_length = -1

    splits = []
    numerical_splits = []
        
    while ptr < len(sequence):
        # Don't linger in one position for too long
        min_length_limit = min_masked_segment \
            if prev_masked_length != 0 else max(min_masked_segment, 1)
        mask_length = randint_functor(min_length_limit, max_masked_segment)

        splits.append(data_processing.split_sequence(
            sequence=sequence,
            mask_start=ptr,
            mask_length=mask_length,
            special_tokens=special_tokens,
        ))

        numerical_splits.append((ptr, mask_length))

        ptr = ptr + mask_length

    return splits, numerical_splits


def get_unique_sequences(df: pandas.DataFrame, field_name: str = "SpikeMutations") -> list:
    unique_proteins = set()
    unique_proteins_list = list()

    for row in df.itertuples():
        prot = getattr(row, field_name)
        lineage = getattr(row, "PangoLineage")
        if (prot, lineage) not in unique_proteins:
            unique_proteins_list.append((prot, lineage))
            unique_proteins.add((prot, lineage))

    return unique_proteins_list


def pango_splitter(proteins: list, frac_train: int, min_val_pangos: int = 1) -> tuple:
    if frac_train >= 1:
        raise ValueError("Train Fraction is too high")

    pango_grouping = defaultdict(list)

    for p, l in proteins:
        pango_grouping[l].append(p)

    pangos = list(pango_grouping.keys())
    n_pango_val = max(1, math.ceil(len(pangos) * (1 - frac_train)))
    random.shuffle(pangos)
    tsequences = []
    vsequences = []
    for tpango in pangos[:-n_pango_val]:
        tsequences.extend(pango_grouping[tpango])
    for vpango in pangos[-n_pango_val:]:
        vsequences.extend(pango_grouping[vpango])
    return tsequences, vsequences


def get_data_splits(
    df_name: str,
    reference: str,
    last_train_date: datetime.datetime,
    last_val_date: Optional[datetime.datetime] = None,
    protein: str = "Spike",
    frac_train: float = 0.9,
    min_val_pangos: int = 1,
    sort_field: str = "Submission date",
    reconstruction_filter: bool = False,
):
    """
    Obtain training data from a pandas DataFrame. If last_val_date is given,
    split based on date (val set is between last_train_date and last_val_date).
    Otherwise split based on pango lineage
    """
    logger.info("Reading DataFrame")
    field_name = f"{protein}Mutations"
    df = read_data_frame_cached(df_name, datefield=sort_field, protein=protein)
    assert(not df.empty)
    if reconstruction_filter:
        df = df[df[f"reconstruction_success_{protein}"]]
        assert(not df.empty)
        assert(len(df) > 0), "Cannot find any sequences after filtration"

    logger.info(f"After reconstruction filtering, found {len(df)} items")
    logger.info("Collecting examples within date %s" % last_train_date.strftime("%Y-%m-%d"))
    df_train = df[df["ParsedDate"] <= last_train_date]
    assert(not df_train.empty)
    assert(len(df_train) > 0), "No training sequences found before date"
    proteins = get_unique_sequences(df_train, field_name=field_name)

    logger.info("Collecting validation examples")

    if last_val_date is not None:
        if last_val_date <= last_train_date:
            raise ValueError("Validation date must be later than training date")

        df_val = df[(df["ParsedDate"] > last_train_date) & (df["ParsedDate"] <= last_val_date)]
        assert(len(df_val) > 0), "No validation sequences found"
        val_proteins = get_unique_sequences(df_val, field_name=field_name)
        train_sequences = set(x[0] for x in proteins)
        val_sequences = [x[0] for x in val_proteins if x[0] not in train_sequences]
    else:
        train_sequences, val_sequences = pango_splitter(
            proteins, frac_train=frac_train, min_val_pangos=min_val_pangos)

    get_full_sequence_functor = partial(get_full_sequence, reference=reference)

    return [(x, get_full_sequence_functor(x)) for x in train_sequences if "stop" not in x], [
        (y, get_full_sequence_functor(y)) for y in val_sequences if "stop" not in y]


def tokenize_sequences(
    sequences: Union[set, list],
    mapper: dict,
    min_masked_segment: int = 0,
    max_masked_segment: int = 128,
    randint_functor: Callable = random.randint,
    randsample_functor: Callable = random.sample,
    msg: str = "Tokenizing",
    num_randomizations_per_sequence: int = 1,
    min_random_segments_per_seq: int = 4,
    select_segments: bool = False,
) -> Generator[tuple, None, None]:
    for seq in tqdm.tqdm(sequences, desc=msg):
        seq_mutation_positions = mutation_positions_in_seq(seq[0])

        all_splits, all_numerical_splits = [], []

        for _1 in range(num_randomizations_per_sequence):
            splits, numerical_splits = create_examples_for_sequence(
                seq[1],
                min_masked_segment=min_masked_segment,
                max_masked_segment=max_masked_segment,
                randint_functor=randint_functor,
            )
            all_splits.extend(splits)
            all_numerical_splits.extend(numerical_splits)

        if not select_segments:
            # Send every sample out
            for enc, dec in all_splits:
                yield [mapper[i] for i in enc], [mapper[j] for j in dec], seq[0]
        else:
            # Make selections so that each mutation position has a representative
            split_selections = []

            location_to_split_mapper = defaultdict(list)
            splits_to_locations_map = defaultdict(list)

            def keyify(split: Union[list, tuple]) -> str:
                x, y = split
                x = "_".join(x)
                y = " ".join(y)
                return f"{x}|{y}"

            for num_split, split in zip(all_numerical_splits, all_splits):
                locations = list(range(num_split[0], sum(num_split)))

                splits_to_locations_map[keyify(split)].extend(locations)

                for i in locations:
                    location_to_split_mapper[i].append(split)

            locations_covered = set()

            for seq_pos_range in seq_mutation_positions:
                for i in range(*seq_pos_range):
                    if i in locations_covered:
                        continue                    
                    split_selection = tuple(randsample_functor(location_to_split_mapper[i], 1).pop())
                    locations_covered = locations_covered.union(
                        splits_to_locations_map[keyify(split_selection)])
                    split_selections.append(split_selection)

            # Now select random partitions of the sequence
            num_random_selections = max(min_random_segments_per_seq, len(split_selections))

            if num_random_selections > 0:
                split_selections.extend(
                    randsample_functor(all_splits, num_random_selections))

            for enc, dec in split_selections:
                yield [mapper[i] for i in enc], [mapper[j] for j in dec], seq[0]


def get_datetime(item: Union[str, None]) -> Union[datetime.datetime, None]:
    if item is None:
        return

    return datetime.datetime.strptime(item, "%Y-%m-%d")


def main(args: argparse.Namespace):
    logger.info("Getting train/val data splits")
    args.last_train_date = get_datetime(args.last_train_date)
    args.last_val_date = get_datetime(args.last_val_date)
    with open(args.ref_file, "r") as fhandle:
        reference = fhandle.read().strip()
    train_sequences, val_sequences = get_data_splits(
        df_name=args.tsv,
        reference=reference,
        last_train_date=args.last_train_date,
        last_val_date=args.last_val_date,
        protein=args.protein_name,
        frac_train=args.frac_train,
        min_val_pangos=args.min_val_pangos,
        sort_field=args.sort_field,
        reconstruction_filter=args.reconstruction_filter,
    )

    logger.info("Tokenizing")

    raw_sequences = {"train": train_sequences, "val": val_sequences}
    mapper = data_processing.Tokenizer().mapper
    num_randomizations = {
        "train": args.num_randomizations_per_sequence_train,
        "val": args.num_randomizations_per_sequence_val}

    sequences = {key: [] for key in raw_sequences}

    for key in raw_sequences:
        for i in range(num_randomizations[key]):
            sequences[key].extend(list(tokenize_sequences(
                raw_sequences[key],
                mapper,
                min_masked_segment=args.min_masked_segment,
                max_masked_segment=args.max_masked_segment,
                randint_functor=args.randint_functor,
                randsample_functor=args.randsample_functor,
                msg=f"Tokenizing {key}, randomization {i + 1}",
                num_randomizations_per_sequence=1,
                min_random_segments_per_seq=args.min_random_segments_per_seq,
                select_segments=args.select_segments
            )))

    logger.info(
        f"Obtained sequence sets %s" % str(
            {key: len(value) for key, value in sequences.items()}))

    logger.info("Initializing disk storage")
    dsets = {key: data_processing.DiskStorage(
        max_length_encoder=max(len(t[0]) for t in sequences[key]) - args.min_masked_segment + 2,
        max_length_decoder=args.max_masked_segment + 1,
        num_sequences=len(sequences[key]),
        datadir=os.path.join(args.datadir, "0", key),
        mode="w+",
    ) for key in sequences}

    for key in sequences:
        logger.info(f"Writing {key} sequences")
        for enc, dec, metadata in sequences[key]:
            dsets[key].append(encoder_data=enc, decoder_data=dec, metadata=metadata)

    for d in dsets.values():
        logger.info(f"Closing dataset of size {len(d)}")
        d.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SARS-CoV2 data")

    parser.add_argument("--tsv", help="Variant surveillance TSV", required=True)
    parser.add_argument("--last_train_date", help="Cut-off date for train set", required=True)
    parser.add_argument(
        "--last_val_date", help=(
            "Cut-off date for val set (if not provided, splits will be made "
            "based on Pango lineages"
        ),
        required=False,
    )
    parser.add_argument("--protein_name", help="Protein name in TSV", default="Spike")
    parser.add_argument(
        "--frac_train", help="Fraction (Pango) for training", type=float, default=0.9)
    parser.add_argument("--min_val_pangos", help="Minimum Pango lineages for val", type=int,
        default=int)
    parser.add_argument("--datadir", help="Output directory", required=True)
    parser.add_argument(
        "--min_masked_segment", help="Minimum masked segment", type=int, default=0)
    parser.add_argument(
        "--max_masked_segment", help="Maximum masked segment", type=int, default=128)
    parser.add_argument(
        "--ref_file", help="Reference file", required=True)
    parser.add_argument(
        "--num_randomizations_per_sequence_train", help="Number of splits per sequence in the training set",
        default=1, type=int)
    parser.add_argument(
        "--num_randomizations_per_sequence_val", help="Number of splits per sequence in the training set",
        default=1, type=int)
    parser.add_argument(
        "--min_random_segments_per_seq", help="Number of random segments per sequence",
        default=4, type=int)
    parser.add_argument(
        "--select_segments", help="Whether to select sampled segments (i.e. not use all of them)",
        default=False, action="store_true")
    parser.add_argument(
        "--sort_field", help="Date field to use for sorting", default="Submission date")
    parser.add_argument(
        "--reconstruction_filter", help="Filter sequences that cannot be reconstructed",
        default=False, action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.randint_functor = random.randint
    args.randsample_functor = random.sample

    main(args)
