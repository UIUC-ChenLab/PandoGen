# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pandas_utils
import json
import datetime
from utils import get_full_sequence
import logging
from typing import List, Set, Union, Generator, Tuple, Optional
import pandas
import argparse
from matplotlib import pyplot as plt
from collections import defaultdict
import tqdm
import pickle
from prediction_tools import MutationResult, CompareType
import os
from dataclasses import asdict
import math

logger = logging.getLogger(__file__)


def has_extra_indel(parent: str, child: str) -> bool:
    parent_mutations = parent.split(",")
    child_mutations = child.split(",")
    return any(c not in parent for c in child_mutations if (("del" in c) or ("ins" in c)))


def read_json_predictions(
    filename: str,
    exclude_indels: bool = False,
    sequence_prevalence: Optional[dict] = None,
    compare_type: CompareType = CompareType.LL,
) -> Generator[Tuple[str, MutationResult], None, None]:
    with open(filename, "r") as fhandle:
        for line in fhandle:
            results = json.loads(line)
            orig_sequence = results["orig_sequence"]
            generations = results["generations"]
            parsed_generations = []
            for j, gen_ in enumerate(generations[1:]):
                gen = [MutationResult(**g, compare_type=compare_type) for g in gen_]
                parsed_generation = []
                root_ll = 0

                if j == 0 and sequence_prevalence:
                    assert(all(g.parent == gen[0].parent for g in gen)), (
                        f"Parent mutation sequence is not the same for all sequences in the first generation, {gen}"
                    )
                    if gen[0].compare_type is CompareType.LL_RATIO:
                        raise ValueError("Cannot use sequence prevalence for LL_RATIO generation")

                    parent_of_gen = gen[0].parent
                    parent_mut = gen[0].parent_mutation_repr
                    if parent_of_gen in sequence_prevalence:
                        root_ll = math.log(sequence_prevalence[parent_of_gen])
                    else:
                        raise ValueError(f"Parent {(parent_of_gen, parent_mut)} not found, possible alignment mismatch, skipping ...")

                for i, m in enumerate(gen):
                    if sequence_prevalence:
                        m.metric += root_ll
                    if (not exclude_indels) or (len(m.parent) == len(m.child)):
                        parsed_generation.append(m)
                    else:
                        logger.debug(f"Skipping {(m.parent, m.child)} due to indels")

                parsed_generations.append(parsed_generation)

            yield orig_sequence, parsed_generations


def process_results(
    filename: str,
    exclude_indels: bool = False,
    sequence_prevalence: Optional[dict] = None,
    compare_type: CompareType = CompareType.LL,
) -> List[dict]:
    logger.info("Reading predictions and collating")

    full_generational_dicts = []
    total = 0

    for orig_sequence, generations in tqdm.tqdm(
        read_json_predictions(
            filename,
            exclude_indels=exclude_indels,
            sequence_prevalence=sequence_prevalence,
            compare_type=compare_type,
        ),
        desc="Collating predictions"
    ):
        for i, gen in enumerate(generations):
            while len(full_generational_dicts) < i + 1:
                full_generational_dicts.append(defaultdict(lambda: MutationResult(None, None)))

            for seq in gen:
                parent = seq.parent_mutation_repr
                full_generational_dicts[i][seq.child] = max(
                    full_generational_dicts[i][seq.child], seq)
                total += 1

    logging.info(f"Obtained {total} sequences")

    return full_generational_dicts


def collapse_generations(generational_dicts: List[dict]) -> dict:
    full_dict = defaultdict(lambda: MutationResult(None, None))

    for gen_dict in generational_dicts:
        for key in gen_dict:
            full_dict[key] = max(full_dict[key], gen_dict[key])

    return full_dict


def parse_date(datestr: str) -> datetime.datetime:
    return datetime.datetime.strptime(datestr, "%Y-%m-%d")


def get_discovery_dates(df: pandas.DataFrame, protein: str = "Spike") -> list:
    discovery_dates = defaultdict(lambda: parse_date("9999-12-31"))
    logger.info("Getting first date of collection")

    for item in tqdm.tqdm(df.itertuples(), desc="Reading dates"):
        mutations = getattr(item, f"{protein}Mutations")
        discovery_dates[mutations] = min(discovery_dates[mutations], item.ParsedDate)

    return discovery_dates


def prepare_plot_data(seq_dict: dict, new_sequences: set, exclusions: set):
    logger.info("Creating plot data")

    seq_dict = {k: v for k, v in seq_dict.items() if k not in exclusions}
    seq_labels = {seq: seq in new_sequences for seq in seq_dict}
    sorted_sequences = sorted(seq_dict.items(), key=lambda x: x[1], reverse=True)
    tp = 0
    fp = 0
    ptr = 0
    plot_pairs = []

    while ptr < len(sorted_sequences):
        s, mut_result = sorted_sequences[ptr]
        metric = mut_result.metric

        if seq_labels[s]:
            tp += 1
        else:
            fp += 1

        ptr += 1
        
        while ptr < len(sorted_sequences) and sorted_sequences[ptr][1] == metric:
            s, mut_result = sorted_sequences[ptr]
            metric = mut_result.metric
            if seq_labels[s]:
                tp += 1
            else:
                fp += 1
            ptr += 1

        plot_pairs.append((fp, tp, metric))

    return plot_pairs


def get_true_false_sequences(predictions: dict, new_sequences: set, exclusions: set) -> tuple:
    true_predictions = []
    false_predictions = []
    all_predictions = []

    for key in predictions:
        if key in exclusions:
            continue
        result = predictions[key]
        if key in new_sequences:
            true_predictions.append(result)
        else:
            false_predictions.append(result)
        all_predictions.append(result)

    def sorter(array: list) -> list:
        return sorted(array, key=lambda x: x.metric, reverse=True)

    return sorter(all_predictions), sorter(true_predictions), sorter(false_predictions)


def calc_sequence_prevalence(df: pandas.DataFrame, ref: str, protein: str = "Spike") -> dict:
    df = df.groupby(f"{protein}Mutations", as_index=False)["Accession ID"].count()
    df = df.rename(columns={"Accession ID": "counts"})
    total = df["counts"].sum()
    df = df.assign(Probability=lambda x: x.counts / total)
    sequence_prevalence = {}
    for k, v in zip(df[f"{protein}Mutations"].tolist(), df["Probability"].tolist()):
        sequence_prevalence[get_full_sequence(k, ref)] = v
    logger.info(f"Sequence prevalence for {len(sequence_prevalence)} items calculated")
    logger.info(f"Type of key = {next(iter(sequence_prevalence.keys()))}")
    return sequence_prevalence


def main(args):
    logger.info("Starting")

    with open(args.ref) as fhandle:
        ref = fhandle.read().strip()

    logger.info("Reading TSV")
    df = pandas_utils.read_data_frame_cached(
        args.tsv, datefield=args.sort_field, protein=args.protein_name)

    tsvbase = os.path.splitext(args.tsv)[0]
    dates_cached = tsvbase + ".start_dates.cached.pkl"
    if os.path.exists(dates_cached):
        with open(dates_cached, "rb") as fhandle:
            discovery_dates = pickle.load(fhandle)
    else:
        discovery_dates = get_discovery_dates(df, protein=args.protein_name)
        with open(dates_cached, "wb") as fhandle:
            pickle.dump({key: value for key, value in discovery_dates.items()}, fhandle)

    last_date = parse_date(args.last_date)

    new_sequences = set(
        get_full_sequence(key, ref) for key, value in discovery_dates.items() if value > last_date)

    exclusions = set(
        get_full_sequence(key, ref) for key, value in discovery_dates.items() if value <= last_date)

    if args.use_sequence_prevalence:
        sequence_prevalence = calc_sequence_prevalence(
            df[df.ParsedDate <= last_date], ref, protein=args.protein_name)
    else:
        sequence_prevalence = None

    generational_predictions = process_results(args.samples, exclude_indels=args.ignore_indels, sequence_prevalence=sequence_prevalence, compare_type=args.compare_type)
    flat_predictions = collapse_generations(generational_predictions)

    flat_plot_pairs = prepare_plot_data(flat_predictions, new_sequences, exclusions)
    generational_plot_pairs = [
        prepare_plot_data(g, new_sequences, exclusions) for g in generational_predictions]

    with open(args.output_prefix + ".pkl", "wb") as fhandle:
        pickle.dump({
            "flat_plot_pairs": flat_plot_pairs,
            "generational_plot_pairs": generational_plot_pairs,
        }, fhandle)

    all_predictions, true_predictions, false_predictions = get_true_false_sequences(
        flat_predictions, new_sequences, exclusions)

    def line_by_line_writer(array: list, filename: str) -> None:
        with open(filename, "w") as fhandle:
            for line in array:
                fhandle.write(json.dumps(line.as_json_dict()) + "\n")

    line_by_line_writer(all_predictions, args.output_prefix + ".all_predictions.json")
    line_by_line_writer(true_predictions, args.output_prefix + ".true_predictions.json")
    line_by_line_writer(false_predictions, args.output_prefix + ".false_predictions.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post process generated results")

    parser.add_argument("--tsv", help="Variants TSV file", required=True)
    parser.add_argument("--last_date", help="Last date of generation sources", required=True)
    parser.add_argument("--samples", help="Generated samples", required=True)
    parser.add_argument("--output_prefix", help="Prefix of output files", required=True)
    parser.add_argument("--protein_name", help="Protein name", default="Spike")
    parser.add_argument("--ref", help="Reference sequence", required=True)
    parser.add_argument("--sort_field", help="Date field to use for sorting", default="Submission date")
    parser.add_argument(
        "--ignore_indels",
        help="Ignore indel type variants",
        required=False,
        action="store_true",
    )
    parser.add_argument("--use_sequence_prevalence", default=False, action="store_true")
    parser.add_argument(
        "--compare_type", help="Comparison type", default="LL", choices=["LL", "LL_RATIO", "PRE_EXISTING"])

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    if args.compare_type == "LL":
        args.compare_type = CompareType.LL
    elif args.compare_type == "PRE_EXISTING":
        args.compare_type = CompareType.PRE_EXISTING
    else:
        args.compare_type = CompareType.LL_RATIO

    main(args)
