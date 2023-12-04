# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pickle
from collections import namedtuple, defaultdict
import matplotlib
from matplotlib import pyplot as plt
import json
import sys
sys.path.append("/home/aramach4/COVID19/covid19")
from find_distance import ScoredData
import os
from typing import List, Optional
matplotlib.rcParams.update({'font.size': 22})
import scipy.stats
import pandas_utils
import datetime
import numpy as np
from plot_results import get95pct_err, uniquify
import pandas
import logging
from argparse import ArgumentParser
import math
import random
from operator import concat
from matplotlib.pyplot import Axes, Figure

logger = logging.getLogger(__file__)

random.seed(42)


def parse_date(date_str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")


def get_sequences_week_range(
    df: pandas.DataFrame,
    start: datetime.datetime,
    end: datetime.datetime
) -> list:
    return set(df[(df.ParsedDate >= start) & (df.ParsedDate < end)].SpikeMutations.tolist())


def get_lineages_week_range(
    df: pandas.DataFrame,
    start: datetime.datetime,
    end: datetime.datetime,
) -> set:
    return set(df[(df.ParsedDate >= start) & (df.ParsedDate < end)].PangoLineage.tolist())


def clamp(x: float, eps: float = 1e-12) -> float:
    return max(x, eps)


def get_sequences_from_exp(
    exp_results: list,
    sequences_nth_week: dict,
) -> dict:
    weekly_data = defaultdict(int)

    for item in exp_results:
        if item.new_seq:
            for key, value in sequences_nth_week.items():
                if item.mutation_repr in value:
                    weekly_data[key] += 1

    return {key: value for key, value in weekly_data.items()}


def get_lineages_from_exp(
    exp_results: list,
    lineages_nth_week: dict,
) -> dict:
    weekly_data = defaultdict(set)

    for item in exp_results:
        if item.new_seq:
            for key, value in lineages_nth_week.items():
                if item.lineage in value:
                    weekly_data[key].add(item.lineage)

    return {key: len(value) for key, value in weekly_data.items()}


def get_stats_for_plot_one_tool(weekly_results: List[dict], n_weeks: int) -> dict:
    results = {}

    for i in range(1, n_weeks + 1):
        i_th_values = [weekly_results[j].get(i, 0) for j in range(len(weekly_results))]
        mean = np.mean(i_th_values)
        err = get95pct_err(i_th_values)
        results[i] = (mean, err)

    return results


def get_cumulative(array: list, n_weeks: int):
    results = []

    for a in array:
        new_result = {}
        for key in range(1, n_weeks + 1):
            new_result[key] = new_result.get(key - 1, 0) + a.get(key, 0)
        results.append(new_result)

    return results


def plot_helper(
    weekly_pandogen: dict,
    weekly_prot_gpt2: dict,
    n_weeks: int,
    output_prefix: Optional[str] = None,
    y_axis: str = "Novel sequences/week",
    full_set: Optional[dict] = None,
    ax: Axes = None,
    savefig: bool = False,
) -> None:
    if full_set:
        full_items = {i: full_set.get(i, 0) for i in range(1, n_weeks + 1)}

        weekly_pandogen = [
            {
                i: weekly.get(i, 0) / clamp(full_items[i]) for i in full_items
            } for weekly in weekly_pandogen
        ]

        weekly_prot_gpt2 = [
            {
                i: weekly.get(i, 0) / clamp(full_items[i]) for i in full_items
            } for weekly in weekly_prot_gpt2
        ]


    pandogen_consolidated = get_stats_for_plot_one_tool(weekly_pandogen, n_weeks)
    prot_gpt2_consolidated = get_stats_for_plot_one_tool(weekly_prot_gpt2, n_weeks)

    logger.info("Plotting results")

    if ax is None:
        plt.figure(figsize=(20, 7))
        ax = plot.subplot()
        savefig = True

    ax.errorbar(
        list(range(1, n_weeks + 1)),
        [pandogen_consolidated[i][0] for i in range(1, n_weeks + 1)],
        yerr=[pandogen_consolidated[i][1] for i in range(1, n_weeks + 1)],
        fmt="go-",
        alpha=0.5,
        capsize=3,
        label="PandoGen",
    )

    ax.errorbar(
        list(range(1, n_weeks + 1)),
        [prot_gpt2_consolidated[i][0] for i in range(1, n_weeks + 1)],
        yerr=[prot_gpt2_consolidated[i][1] for i in range(1, n_weeks + 1)],
        fmt="rx-",
        alpha=0.5,
        capsize=3,
        label="Prot GPT2",
    )

    ax.set_xticks(list(range(1, n_weeks + 1, 3)))
    ax.set_xticklabels(list(range(1, n_weeks + 1, 3)))
    ax.set_xlabel("Weeks after training")
    ax.set_ylabel(y_axis)

    if savefig:
        logger.info("Saving figure")
        plt.legend(loc="upper left")
        plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")


def saliency_fraction_helper(a: dict, b: dict) -> dict:
    result = {key: b.get(key, 0) / clamp(a[key]) for key in a}
    return result


def merge_results(results: dict, key_prefixes: List[str], n_merge: int) -> dict:
    new_results = {}

    for key_prefix in key_prefixes:
        keys = [key for key in results if os.path.abspath(key).startswith(key_prefix)]
        if len(keys) % n_merge != 0:
            raise ValueError(f"{len(keys)} items cannot be merged into {n_merge} items")
        random.shuffle(keys)
        splits = [x.tolist() for x in np.split(np.array(keys), indices_or_sections=n_merge, axis=0)]

        logger.info(f"Merging {key_prefix} as follows: {splits}")

        for i, split in enumerate(splits):
            new_results[f"{key_prefix}/{i}"] = concat(*[results[s] for s in split])

    return new_results


def main(args):
    logger.info("Reading data")
    df = pandas_utils.read_data_frame_cached(
        args.tsv,
        protein="Spike",
        datefield="Submission date",
    )

    df_counts = df.groupby("SpikeMutations", as_index=False)["Accession ID"].count()
    salient = set(
        df_counts[df_counts["Accession ID"] >= args.count_threshold].SpikeMutations.tolist())

    orig_df = df

    df = df.loc[df.groupby("SpikeMutations").ParsedDate.idxmin()]
    df_dates = dict(zip(df.SpikeMutations, df.ParsedDate))

    df_pango_dates = orig_df.loc[orig_df.groupby("PangoLineage").ParsedDate.idxmin()]
    pango_dates = dict(zip(df_pango_dates.PangoLineage, df_pango_dates.ParsedDate))

    last_date = parse_date(args.last_date)
    start = last_date + datetime.timedelta(days=1)

    logger.info("Getting sequences discovered in each week")
    sequences_nth_week = {
        i: get_sequences_week_range(
            df, start + datetime.timedelta(7 * (i - 1)), start + datetime.timedelta(7 * i)
        ) for i in range(1, args.n_weeks + 1)
    }
    salient_sequences_nth_week = {
        key: set(value).intersection(salient) for key, value in sequences_nth_week.items()
    }

    lineages_nth_week = {
        i: get_lineages_week_range(
            df_pango_dates, start + datetime.timedelta(7 * (i - 1)), start + datetime.timedelta(7 * i)
        ) for i in range(1, args.n_weeks + 1)
    }

    logger.info("Number of salient sequences: "
        f"{[(key, len(value)) for key, value in salient_sequences_nth_week.items()]}"
    )

    logger.info("Reading evaluation results")

    args.pandogen_prefix = f"{os.path.abspath(args.pandogen_prefix)}/"
    args.prot_gpt2_prefix = f"{os.path.abspath(args.prot_gpt2_prefix)}/"

    with open(args.eval_results, "rb") as fhandle:
        results = pickle.load(fhandle)

        if args.n_merge:
            logger.info(f"Merging results into {args.n_merge} result sets")
            results = merge_results(
                results,
                [
                    args.pandogen_prefix,
                    args.prot_gpt2_prefix,
                ],
                n_merge=args.n_merge,
            )

    weekly_results_pandogen = []
    weekly_results_prot_gpt2 = []

    weekly_lineage_results_pandogen = []
    weekly_lineage_results_prot_gpt2 = []

    weekly_results_salient_pandogen = []
    weekly_results_salient_prot_gpt2 = []

    weekly_results_saliency_fraction_pandogen = []
    weekly_results_saliency_fraction_prot_gpt2 = []

    logger.info("Getting plot values for weekly discoveries")

    for key in results:
        key_abs = os.path.abspath(key)

        if key_abs.startswith(args.pandogen_prefix):
            uniquified_results = uniquify(results[key])

            salient_results = [x for x in uniquified_results if x.new_seq and x.count >= args.count_threshold]
            dates = [df_dates[x.mutation_repr] for x in salient_results]

            weekly_results_pandogen.append(
                get_sequences_from_exp(uniquified_results, sequences_nth_week)
            )

            weekly_results_salient_pandogen.append(
                get_sequences_from_exp(uniquified_results, salient_sequences_nth_week)
            )

            weekly_results_saliency_fraction_pandogen.append(
                saliency_fraction_helper(
                    weekly_results_pandogen[-1],
                    weekly_results_salient_pandogen[-1]
                )
            )

            weekly_lineage_results_pandogen.append(
                get_lineages_from_exp(uniquified_results, lineages_nth_week)
            )

        if key_abs.startswith(args.prot_gpt2_prefix):
            uniquified_results = uniquify(results[key])

            weekly_results_prot_gpt2.append(
                get_sequences_from_exp(uniquified_results, sequences_nth_week)
            )

            weekly_results_salient_prot_gpt2.append(
                get_sequences_from_exp(uniquified_results, salient_sequences_nth_week)
            )

            weekly_results_saliency_fraction_prot_gpt2.append(
                saliency_fraction_helper(
                    weekly_results_prot_gpt2[-1],
                    weekly_results_salient_prot_gpt2[-1]
                )
            )

            weekly_lineage_results_prot_gpt2.append(
                get_lineages_from_exp(uniquified_results, lineages_nth_week)
            )

    lineages_nth_week = {key: len(value) for key, value in lineages_nth_week.items()}

    if args.cumulative:
        weekly_results_pandogen = get_cumulative(weekly_results_pandogen, args.n_weeks)
        weekly_results_prot_gpt2 = get_cumulative(weekly_results_prot_gpt2, args.n_weeks)
        weekly_results_salient_pandogen = get_cumulative(weekly_results_salient_pandogen, args.n_weeks)
        weekly_results_salient_prot_gpt2 = get_cumulative(weekly_results_salient_prot_gpt2, args.n_weeks)

        weekly_results_saliency_fraction_pandogen.clear()
        weekly_results_saliency_fraction_prot_gpt2.clear()

        weekly_results_saliency_fraction_pandogen = [
            saliency_fraction_helper(a, b) for a, b in zip(weekly_results_pandogen, weekly_results_salient_pandogen)
        ]

        weekly_results_saliency_fraction_prot_gpt2 = [
            saliency_fraction_helper(a, b) for a, b in zip(weekly_results_prot_gpt2, weekly_results_salient_prot_gpt2)
        ]

        weekly_lineage_results_pandogen = get_cumulative(weekly_lineage_results_pandogen, args.n_weeks)
        weekly_lineage_results_prot_gpt2 = get_cumulative(weekly_lineage_results_prot_gpt2, args.n_weeks)

        lineages_nth_week = get_cumulative([lineages_nth_week], args.n_weeks)[0]

    plt.figure(figsize=(13, 10))
    matplotlib.rcParams.update({'font.size': 12})

    plot_helper(
        weekly_results_pandogen,
        weekly_results_prot_gpt2,
        args.n_weeks,
        y_axis="#(novel,real)",
        ax=plt.subplot(221),
    )

    plot_helper(
        weekly_results_salient_pandogen,
        weekly_results_salient_prot_gpt2,
        args.n_weeks,
        y_axis="#(salient,novel,real)",
        ax=plt.subplot(222),
    )

    plot_helper(
        weekly_results_saliency_fraction_pandogen,
        weekly_results_saliency_fraction_prot_gpt2,
        args.n_weeks,
        y_axis="#(salient,novel,real)/#(novel,real)",
        ax=plt.subplot(223),
        savefig=False,
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    plt.savefig(f"{args.output_prefix}.salience.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 10))

    plot_helper(
        weekly_lineage_results_pandogen,
        weekly_lineage_results_prot_gpt2,
        args.n_weeks,
        y_axis="#novel forecasted lineages",
        ax=plt.subplot(211),
    )

    plot_helper(
        weekly_lineage_results_pandogen,
        weekly_lineage_results_prot_gpt2,
        args.n_weeks,
        y_axis="#novel forecasted lineages/#reported lineages",
        full_set=lineages_nth_week,
        ax=plt.subplot(212),
        savefig=False,
    )

    plt.legend(loc="upper left")
    plt.savefig(f"{args.output_prefix}.lineage.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot weekly results")

    parser.add_argument(
        "--tsv",
        help="Path to TSV file",
        required=True,
    )

    parser.add_argument(
        "--output_prefix",
        help="Prefix of output file",
        required=True,
    )

    parser.add_argument(
        "--last_date",
        help="Last date of training",
        required=True,
    )

    parser.add_argument(
        "--n_weeks",
        help="Number of weeks to look forward",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--eval_results",
        help="Evaluation results",
        required=True,
    )

    parser.add_argument(
        "--pandogen_prefix",
        help="Prefix of pandogen experiment",
        required=True,
    )

    parser.add_argument(
        "--prot_gpt2_prefix",
        help="Prefix of prot gpt2 experiment",
        required=True,
    )

    parser.add_argument(
        "--count_threshold",
        help="Threshold of counts for TP/FN calculation",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--n_merge",
        help="Merge eval results into n_merge items, if provided",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--cumulative",
        help="Accumulate over weeks",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    main(args)
