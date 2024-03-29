# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pandas_utils
import datetime
from utils import get_full_sequence, ambiguity_mapping, _AMBIGUOUS_CHARACTERS
import json
import pickle
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Union, Dict
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import tqdm
import binary_search_parameter
from collections import defaultdict


@dataclass
class ScoredData:
    new_seq: bool
    old_seq: bool
    false_seq: bool
    invalid_seq: bool
    score: float
    seq: str
    count: int
    mutation_repr: Optional[str] = None
    min_seq: Optional[str] = None
    min_dist: Optional[int] = None
    seq_date: Optional[datetime.datetime] = None
    min_seq_date: Optional[datetime.datetime] = None
    novel_kmers: Optional[int] = None
    kmer_length: Optional[int] = None
    lineage: Optional[str] = None
    new_pango_flag: Optional[bool] = None

    def __post_init__(self):
        if self.invalid_seq:
            self.new_seq = False
            self.old_seq = False
            self.false_seq = False
            self.new_pango_flag = False


class SeqChecker:
    def __init__(
        self,
        seq_set: Union[set, dict],
        **checker_dicts,
    ):
        self.seq_set = seq_set
        self.seq_set_by_length = defaultdict(set)
        for seq in seq_set:
            self.seq_set_by_length[len(seq)].add(seq)
        self.cached_results = dict()
        self.checker_dicts = checker_dicts

    def __call__(self, seq_to_compare: str):
        seq_length = len(seq_to_compare)
        other_return_values = None

        if seq_to_compare in self.cached_results:
            return self.cached_results[seq_to_compare]

        elif seq_to_compare in self.seq_set:
            result = True
            other_return_values = {
                key: self.checker_dicts[key].get(seq_to_compare, None)
                for key in self.checker_dicts
            }
        elif seq_length not in self.seq_set_by_length:
            result = False

        else:
            comparands = self.seq_set_by_length[seq_length]

            for c in comparands:
                for a, b in zip(seq_to_compare, c):
                    if a == b or "X" in [a, b]:
                        continue

                    if b in ambiguity_mapping.get(a, []):
                        continue

                    if a in ambiguity_mapping.get(b, []):
                        continue

                    break
                else:
                    result = True
                    other_return_values = {
                        key: self.checker_dicts[key].get(c, None)
                        for key in self.checker_dicts
                    }
                    break
            else:
                result = False

        self.cached_results[seq_to_compare] = (result, other_return_values)

        return result, other_return_values


def parse_date(x: str) -> datetime.datetime:
    return datetime.datetime.strptime(x, "%Y-%m-%d")


def read_data(
    tsv: str,
    last_date: str,
    ref: str,
    datefield: str = "Submission date",
    protein: str= "Spike",
):
    df = pandas_utils.read_data_frame_cached(
        tsv,
        datefield=datefield,
        protein=protein,
    )

    df_counts = df.groupby("SpikeMutations", as_index=False)["Accession ID"].count()
    count_dict = dict(zip(df_counts.SpikeMutations.tolist(), df_counts["Accession ID"].tolist()))

    df = df.loc[df.groupby("SpikeMutations").ParsedDate.idxmin()]

    # Removing this as it is not used when creating training data in
    # random_split_fasta.py. It is, however used in create_competition_validation_data
    # and create_occurrence_buckets, which is okay
    # df = df[df[f"reconstruction_success_{protein}"]]

    old_df = df[df.ParsedDate <= parse_date(last_date)]
    new_df = df[df.ParsedDate > parse_date(last_date)]
    new_df_date_dict = {
        get_full_sequence(s, ref): d for s, d in zip(
            new_df.SpikeMutations.tolist(), new_df.ParsedDate.tolist())
    }

    df_pango = df.loc[df.groupby("PangoLineage").ParsedDate.idxmin()]

    pango_dict = defaultdict(list)

    for s, p in zip(new_df.SpikeMutations.tolist(), new_df.PangoLineage.tolist()):
        full_seq = get_full_sequence(s, ref)
        pango_dict[full_seq].append(p)

    new_pangos = set(df_pango[df_pango.ParsedDate > parse_date(last_date)].PangoLineage.tolist())

    old_mutations = old_df.SpikeMutations.tolist()
    new_mutations = new_df.SpikeMutations.tolist()

    old_sequences = [get_full_sequence(i, ref) for i in old_mutations]
    old_sequences_with_counts = {}

    for m, s in zip(old_mutations, old_sequences):
        old_sequences_with_counts[s] = count_dict[m]

    new_sequences = [get_full_sequence(j, ref) for j in new_mutations]
    new_sequences_with_counts = {}

    for m, s in zip(new_mutations, new_sequences):
        new_sequences_with_counts[s] = count_dict[m]

    mutation_map = {get_full_sequence(j, ref): j for j in new_mutations}

    return (
        old_sequences_with_counts,
        new_sequences_with_counts,
        pango_dict,
        mutation_map,
        new_pangos,
        new_df_date_dict,
    )


def is_invalid(seq: str):
    # Special character in SDA/PandoGen
    if "[" in seq or "]" in seq:
        return True

    # Special character in Prot-GPT2
    if "<" in seq or ">" in seq:
        return True

    # End of sequence in all cases
    if seq.count("*") != 1:
        return True


def process_file(
    prediction_file: str,
    old_seq: dict,
    new_seq: dict,
    kmer_length: int = 11,
    pango_dict: Optional[dict] = None,
    mut_map: Optional[dict] = None,
    new_pangos: Optional[set] = None,
    new_df_date_dict: Optional[set] = None,
):
    old_seq_checker = SeqChecker(old_seq)
    new_seq_checker = SeqChecker(
        new_seq,
        mutation_map=mut_map,
        lineage_dict=pango_dict,
        seq_dates_dict=new_df_date_dict
    )
    ref_kmers = binary_search_parameter.get_kmers(old_seq, length=kmer_length)
    result_tuples = []

    with open(prediction_file, "r") as fhandle:
        for line in fhandle:
            items = json.loads(line)
            seq = items["seq"]
            score = float(items[args.key])
            new_flag = False
            old_flag = False
            novel_kmers = None
            lineage = None
            seq_date = None
            muts = None

            invalid_flag = is_invalid(seq)

            if not invalid_flag:
                old_flag, _ = old_seq_checker(seq)
                if not old_flag:
                    new_flag, other_returns = new_seq_checker(seq)
                    if other_returns:
                        lineage = other_returns["lineage_dict"]
                        seq_date = other_returns["seq_dates_dict"]
                        muts = other_returns["mutation_map"]
                novel_kmers = binary_search_parameter.get_per_sequence_novelty(
                    seq, ref_kmers=ref_kmers, known_valid=True)

            new_pango_flag = (lineage is not None) and all(p in new_pangos for p in lineage)

            result_tuples.append(
                ScoredData(
                    new_seq=new_flag,
                    old_seq=old_flag,
                    false_seq=not(old_flag or new_flag),
                    invalid_seq=invalid_flag,
                    score=score,
                    seq=seq,
                    seq_date=seq_date,
                    count=new_seq.get(seq, 0),
                    novel_kmers=novel_kmers,
                    kmer_length=kmer_length,
                    lineage=lineage,
                    mutation_repr=muts,
                    new_pango_flag=new_pango_flag,
                )
            )

    result_tuples = sorted(result_tuples, key=lambda x: x.score, reverse=True)

    return prediction_file, result_tuples


def main(args):
    with open(args.ref) as fhandle:
        ref = fhandle.read().strip()

    logging.info("Reading data")
    (
        old_seq,
        new_seq,
        pango_dict,
        mut_map,
        new_pangos,
        new_df_date_dict,
    ) = read_data(args.tsv, args.last_date, ref, args.datefield, args.protein)
    process_file_functor = partial(
        process_file,
        old_seq=old_seq,
        new_seq=new_seq,
        kmer_length=args.kmer_length,
        pango_dict=pango_dict,
        mut_map=mut_map,
        new_pangos=new_pangos,
        new_df_date_dict=new_df_date_dict,
    )

    logging.info("Preparing plot data")

    result_dict = {}

    files = args.decoder_predictions.split(",")
    workers = Pool(args.num_workers)

    for result_key, result_value in tqdm.tqdm(
        workers.imap_unordered(process_file_functor, files), desc="Processing files", total=len(files)):
        result_dict[result_key] = result_value

    with open(args.output, "wb") as fhandle:
        pickle.dump(result_dict, fhandle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post process generated results")

    parser.add_argument("--tsv", help="Variants TSV file", required=True)
    parser.add_argument("--last_date", help="Last date of generation sources", required=True)
    parser.add_argument("--ref", help="Reference sequence", required=True)
    parser.add_argument("--datefield", help="Date field to use for sorting", default="Submission date")
    parser.add_argument("--protein", help="Protein name", default="Spike")
    parser.add_argument("--decoder_predictions", help="Generated samples (comma-separated files)", required=True)
    parser.add_argument("--key", help="Score to look at", required=True)
    parser.add_argument("--output", help="Output file name", required=True)
    parser.add_argument("--num_workers", help="Number of workers", default=12, type=int)
    parser.add_argument("--kmer_length", help="Kmer-length for novelty", default=11, type=int)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    main(args)

