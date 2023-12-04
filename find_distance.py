# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import json
from Bio import pairwise2
from post_process_decoder_generations import parse_date
import re
import itertools
import mutation_edit_distance
from functools import lru_cache, partial
from typing import Optional, List, Set, Callable
from dataclasses import asdict
import tqdm
import pickle
from argparse import ArgumentParser
import logging
from pandas_utils import read_data_frame_cached
import datetime
from utils import get_full_sequence
import os
import torch.distributed
from post_process_decoder_generations import ScoredData

GAP_PENALITY = -4
GAP_EXTEND_PENALTY = -0.1
MATCH_SCORE = 1
MISMATCH_PENALTY = -1

mutation_edit_distance.DELETION_PENALTY = float("inf")

logger = logging.getLogger(__file__)


def uniquify(l: list) -> list:
    unique_set = set()
    res = []
    for item in l:
        if item.seq not in unique_set:
            res.append(item)
        unique_set.add(item.seq)
        
    return res


def biopython_aligner(seqA, seqB):
    """
    Global alignment of two sequences using Biopython

    :param seqA: str
        Sequence A

    :param seqB: str
        Sequence B
    """
    # alignment = pairwise2.align.globalcs(
    #     seqA, seqB, biopython_scorer, GAP_PENALITY, GAP_EXTEND_PENALTY)
    alignment = pairwise2.align.globalms(
        seqA, seqB, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALITY, GAP_EXTEND_PENALTY)
    return alignment[0]


def resolve_insertion_clusters(mutations: list) -> list:
    result = []

    for item in mutations:
        if isinstance(item, list) and item[0].startswith("ins"):
            aas = "".join([x[-1] for x in item])
            pos = item[0][3: -1]
            result.append(f"ins{pos}{aas}")
        else:
            result.append(item)

    return result


def read_tsv(tsv: str, last_date: str, datefield: str, protein: str) -> tuple:
    df = read_data_frame_cached(
        tsv, datefield=datefield, protein=protein,
    )
    df_dates = df.loc[df.groupby("SpikeMutations").ParsedDate.idxmin()]

    old_mutations = list(set(df[df.ParsedDate <= parse_date(last_date)].SpikeMutations.tolist()))
    new_mutations = list(set(df[df.ParsedDate > parse_date(last_date)].SpikeMutations.tolist()))

    return new_mutations, old_mutations, df_dates


def group_mutations(mutations: list):
    last_item = None
    cluster = []
    results = []

    for m in mutations:
        try:
            item = re.findall(r"([A-Za-z]+)(\d+)([A-Za-z]+)", m)[0]
        except IndexError as e:
            logger.error(f"Cannot process {m}")
            raise e

        item = [item[0], int(item[1]), item[2]]

        if item[0] == "ins":
            if last_item is None or (
                last_item[0] == "ins" and last_item[1] == item[1]):
                cluster.append(m)
            else:
                if cluster:
                    results.append(cluster)
                cluster = [m]

        elif item[-1] == "del":
            if last_item is None or (
                last_item[-1] == "del" and last_item[1] == item[1] - 1):
                cluster.append(m)
            else:
                if cluster:
                    results.append(cluster)
                cluster = [m]

        else:
            if cluster:
                results.append(cluster)
                cluster = []
            results.append(m)

    if cluster:
        results.append(cluster)

    return results


def get_mutation_repr(seq: str, ref: str):
    alignment = biopython_aligner(ref, seq)
    ref_pos = 0
    mutations = []

    for i, (a, b) in enumerate(zip(alignment.seqA, alignment.seqB)):
        if a != b:
            if "-" not in (a, b):
                mutations.append(f"{a}{ref_pos + 1}{b}")
            else:
                if a == "-":
                    mutations.append(f"ins{ref_pos}{b}")
                else:
                    mutations.append(f"{a}{ref_pos + 1}del")

        if a != "-":
            ref_pos += 1

    return mutations


def find_closest_historical_sequence(seq: str, ref: str, old_sequences: list) -> tuple:
    """
    Find the closest historical sequence in terms of mutations
    """
    mutation_repr_grouped = resolve_insertion_clusters(
        group_mutations(get_mutation_repr(seq, ref)))
    mutation_repr = ",".join(
        [x if isinstance(x, str) else ",".join(x) for x in mutation_repr_grouped])
    min_dist = float("inf")
    min_seq = None

    for o in old_sequences:
        d0 = mutation_edit_distance.edit_distance(o, mutation_repr)  # Allow insertion into o
        d1 = -mutation_edit_distance.edit_distance(mutation_repr, o)  # Allow insertion into seq

        if d0 <= -d1:
            min_dist_for_case = d0
        else:
            min_dist_for_case = d1

        """
        If the total number of mutations to be made in a sequence
        is lower than before, accept it. It can be new mutations, or canceling
        older mutations

        If the same number of mutations have to be decided between deleting
        existing mutations and adding new mutations, we choose the deletion case
        as this means the new sequence probably doesn't carry much new information
        """
        if abs(min_dist_for_case) < min_dist:
            min_dist = min_dist_for_case
            min_seq = o
        elif abs(min_dist_for_case) == min_dist and min_dist_for_case < 0:
            min_dist = min_dist_for_case
            min_seq = o

    return min_seq, min_dist, mutation_repr


def get_cached_distance_measure(ref: str, old_sequences: list) -> Callable:
    functor = partial(find_closest_historical_sequence, ref=ref, old_sequences=old_sequences)
    cached_functor = lru_cache(maxsize=None)(functor)
    return cached_functor


def find_closest_for_all_sequences(
    sequences: List[ScoredData],
    ref: str,
    old_sequences: list,
    cached_functor: Optional[Callable] = None,
    disable_tqdm: bool = False,
):
    results = []

    if cached_functor is not None:
        functor = cached_functor
    else:
        functor = partial(
            find_closest_historical_sequence, ref=ref, old_sequences=old_sequences)

    for seq in tqdm.tqdm(sequences, desc="Finding closest sequences", disable=disable_tqdm):
        # if not seq.new_seq:
        #     continue

        if "[" in seq.seq or "]" in seq.seq:
            continue

        if "*" not in seq.seq:
            continue
            
        min_seq, min_dist, mutation_repr = functor(seq.seq)

        arguments = asdict(seq)
        arguments["mutation_repr"] = mutation_repr
        arguments["min_seq"] = min_seq
        arguments["min_dist"] = min_dist

        new_item = ScoredData(**arguments)

        results.append(new_item)

    return results


def main(args):
    logger.info("Loading inputs")

    with open(args.eval_file, "rb") as fhandle:
        items = pickle.load(fhandle)

    logger.info("Loading reference")
        
    with open(args.ref, "r") as fhandle:
        ref = fhandle.read().strip()

    if args.local_rank is None or args.local_rank == 0:
        logger.info("Loading variants")
        new_sequences, old_sequences, df_dates = read_tsv(
            args.tsv,
            args.last_date,
            args.datefield,
            args.protein,
        )
        df_items = [new_sequences, old_sequences, df_dates]
        logger.info("Done loading variants")
    else:
        df_items = [None, None, None]

    if args.local_rank is not None:
        torch.distributed.broadcast_object_list(df_items, src=0)
        if args.local_rank != 0:
            new_sequences, old_sequences, df_dates = df_items

        if args.local_rank == 0:
            logger.info("Done broadcast")

    df_dates_dict = {}

    for m, d in zip(df_dates.SpikeMutations, df_dates.ParsedDate):
        df_dates_dict[get_full_sequence(m, ref)] = d

    old_sequences = [x for x in old_sequences]

    logger.info("Performing mutation alignments")

    all_results = {}

    dist_functor = get_cached_distance_measure(ref=ref, old_sequences=old_sequences)

    keys = items.keys()

    if args.tgt_keys:
        key_prefixes = args.tgt_keys.split(",")
        new_keyset = list()

        for k in keys:
            if any(k.startswith(x) for x in key_prefixes):
                new_keyset.append(k)

        keys = new_keyset

    keys = sorted(keys)

    if args.local_rank is not None:
        new_keys = []

        for i, k in enumerate(keys):
            if (i + args.local_rank) % args.world_size == 0:
                new_keys.append(k)

        keys = new_keys

    for key in keys:
        tgt = uniquify(items[key])
        logger.info(f"Calculating min distance for key {key}")
        all_results[key] = find_closest_for_all_sequences(
            sequences=tgt,
            ref=ref,
            old_sequences=old_sequences,
            cached_functor=dist_functor,
            disable_tqdm=args.local_rank is not None,
        )
        logger.info(f"Assigning dates")
        for seq in all_results[key]:
            if seq.seq in df_dates_dict:
                seq.seq_date = df_dates_dict[seq.seq]
            seq.min_seq_date = df_dates_dict[get_full_sequence(seq.min_seq, ref)]

    logger.info("Writing results")

    output_name = args.output

    if args.local_rank is not None:
        output_prefix, ext = os.path.splitext(args.output)
        output_name = f"{output_prefix}{args.local_rank}{ext}"

    with open(output_name, "wb") as fhandle:
        pickle.dump(all_results, fhandle)


if __name__ == "__main__":
    parser = ArgumentParser(description="Find distance metrics for generated data")

    parser.add_argument("--eval_file", help="Evaluation file", required=True)
    parser.add_argument("--ref", help="Reference file", required=True)
    parser.add_argument("--tsv", help="Variants TSV", required=True)
    parser.add_argument("--last_date", help="Last date of training", required=True)
    parser.add_argument("--datefield", help="Date field", default="Submission date")
    parser.add_argument("--protein",
        help="Protein for which to conduct analysis", default="Spike")
    parser.add_argument("--output", help="Output file", required=True)
    parser.add_argument("--tgt_keys", help="Comma-separated list of keys", required=False)

    args = parser.parse_args()

    try:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(
            world_size=args.world_size, rank=args.local_rank, backend="gloo")
    except KeyError:
        args.local_rank = None
        args.worldsize = None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    main(args)
