# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import json
from plot_results import eq, uniquify, get95pct_err, GlobalStats
import pickle
from post_process_decoder_generations import ScoredData
import os
from collections import defaultdict, namedtuple
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from typing import List, Optional
from enum import Enum, auto
from matplotlib.lines import Line2D
import logging
import numpy as np
import matplotlib
import datetime
import math

font = {'size': 14}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)

logger = logging.getLogger()

LocalStats = namedtuple("LocalStats", ["fp", "tp", "tp_count", "rate"])
DatedStats = namedtuple(
    "DatedStats",
    ["case_counts", "seq_counts", "lineage_counts", "salient_seq_counts", "salient_to_seq_ratio"]
)

matplotlib.rcParams.update({'font.size': 14})

tool_colors = {
    "PandoGen": "g",
    "Prot GPT2 enumerated": "b",
    "Prot GPT2 unenumerated": "r",
    "SDA": "y",
    "ProGen2": "c",
}


class LocalStatType(Enum):
    NOVEL = auto()
    ALL = auto()


def get_global_stats(items: List[ScoredData]) -> tuple:
    """
    Convert a single set of stats for all the results in a file
    """
    total_valid_sequences = 0
    total_new_sequences = 0
    total_false_sequences = 0
    total_old_sequences = 0
    total_kmer_novelty = 0
    total_case_counts = 0
    total_non_old = 0
    total_new_lineages = 0
    total_salient_sequences = 0
    
    seq_set = set()
    lineages_set = set()
    
    for item in items:
        if item.invalid_seq:
            continue
            
        if item.seq in seq_set:
            continue
            
        total_valid_sequences += 1
        total_new_sequences += 1 if item.new_seq else 0
        total_false_sequences += 1 if item.false_seq else 0
        total_old_sequences += 1 if item.old_seq else 0
        total_kmer_novelty += item.novel_kmers
        total_case_counts += item.count if item.new_seq else 0
        total_non_old += 1 if not item.old_seq else 0
        total_salient_sequences += 1 if item.new_seq and item.count >= 10 else 0

        new_unique_lineage_found = item.new_seq and item.new_pango_flag and all(
            x not in lineages_set for x in item.lineage
        )
        
        total_new_lineages += 1 if new_unique_lineage_found else 0

        if item.lineage:
            for l in item.lineage:
                lineages_set.add(l)

        seq_set.add(item.seq)
        
    return GlobalStats(
        n_valid=total_valid_sequences,
        n_new=total_new_sequences,
        n_false=total_false_sequences,
        n_old=total_old_sequences,
        avg_kmer_novelty=total_kmer_novelty / total_valid_sequences,
        old_rate=total_old_sequences / total_valid_sequences,
        new_rate=total_new_sequences / total_valid_sequences,
        ppv=total_new_sequences / max(total_new_sequences + total_false_sequences, 1e-12),
        counts=total_case_counts,
        n_non_old=total_non_old,
        n_new_lineages=total_new_lineages,
        case_count_per_forecast=total_case_counts / total_new_sequences if total_new_sequences > 0 else 0,
        case_count_per_novel=total_case_counts / total_non_old if total_non_old > 0 else 0,
        n_salient_sequences=total_salient_sequences,
    )


def global_stats_plotter(res: dict, output_dir: str, x_axis_key: Optional[str] = None, x_axis_name: Optional[str] = None, marker_append: str = "-", min_non_old: int = None):
    """
    Plot global stats
    """
    font = {'size': 30}
    matplotlib.rc('font', **font)

    fmt_keys = {
        "PandoGen": "o",
        "Prot GPT2 unenumerated": "x",
        "Prot GPT2 enumerated": "s",
        "SDA": "^",
        "ProGen2": "8",
    }

    legend_dict = {
        tool: Line2D(
            [0],
            [0],
            marker=fmt_keys[tool],
            color=tool_colors[tool],
            label=r"\textbf{" + tool + "}" if tool == "PandoGen" else tool,
            markerfacecolor=tool_colors[tool],
            markersize=5
        ) for tool in fmt_keys
    }

    def plot_helper(tool, metric, ax, fmt, y_name):
        mean, yerr = [], []
        x_axis_mean, x_axis_err = [], []

        for top_p in res[tool]:
            if min_non_old and res[tool][top_p]["n_non_old"][0] < min_non_old:
                continue

            mean_, yerr_ = res[tool][top_p][metric]
            mean.append(mean_)
            yerr.append(yerr_)

            if x_axis_key:
                x_axis_mean_, x_axis_err_ = res[tool][top_p][x_axis_key]
                x_axis_mean.append(x_axis_mean_)
                x_axis_err.append(x_axis_err_)

        if not x_axis_key:
            x_axis_mean = [float(x) for x in res[tool].keys()]
            x_axis_err = None

        ax.errorbar(
            x_axis_mean,
            mean,
            fmt=fmt,
            xerr=x_axis_err,
            yerr=yerr,
            alpha=0.75,
            capsize=3,
            label=tool,
        )
        ax.set_xlabel(x_axis_name if x_axis_name else "p (nucleus sampling)")
        ax.set_ylabel(y_name)

    plt.figure(figsize=(30, 30))

    for metric, coords, y_name in zip(
        ["ppv", "counts", "n_new", "n_new_lineages", "avg_kmer_novelty", "n_salient_sequences"],
        [321, 322, 323, 324, 325, 326],
        ["PPV", "Case counts", r"\#Forecasts", r"\#Forecasted lineages", "k-mer distance", r"\#Salient forecasts"],
    ):
        if metric == x_axis_key:
            continue
        ax = plt.subplot(coords)
        for tool in res:
            plot_helper(
                tool,
                metric,
                ax,
                fmt=tool_colors[tool] + fmt_keys[tool] + marker_append,
                y_name=y_name
            )

    legend_elements = [legend_dict[key] for key in sort_tools(list(res.keys()))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.05, -0.25), loc="upper center", fancybox=True, ncol=2)
    plt.savefig(os.path.join(output_dir, "Macro.png"), dpi=300, bbox_inches="tight")

    font = {'size': 14}
    matplotlib.rc('font', **font)


def get_local_stats(results: List[ScoredData], local_stat_type: LocalStatType) -> List[LocalStats]:
    """
    Convert results from a file into a sequence of stats (tp/fp etc)
    """
    local_stats = []

    for item in results:
        if local_stats:
            last_fp, last_tp, last_tp_count, _ = local_stats[-1]
        else:
            last_fp, last_tp, last_tp_count = 0, 0, 0

        if local_stat_type is LocalStatType.NOVEL:
            fp_flag = item.false_seq and (not item.invalid_seq)
        elif local_stat_type is LocalStatType.ALL:
            fp_flag = not item.invalid_seq
        else:
            raise AttributeError(f"Uknown local_stat_type: {local_stat_type}")

        if item.new_seq:
            new_fp = last_fp
            new_tp = last_tp + 1
            new_tp_count = last_tp_count + item.count
        elif fp_flag:
            new_fp = last_fp + 1
            new_tp = last_tp
            new_tp_count = last_tp_count
        else:
            continue

        new_local_stat = LocalStats(
            fp=new_fp,
            tp=new_tp,
            tp_count=new_tp_count,
            rate=new_tp / (new_tp + new_fp),
        )

        local_stats.append(new_local_stat)

    return local_stats


def get_local_stats_dated(results: List[ScoredData], date_cutoff: str, max_week_spec: int = None) -> List[LocalStats]:
    """
    For a date-sorted result set, find out the relevant stats over time
    """
    min_date = datetime.datetime.strptime(date_cutoff, "%Y-%m-%d")
    max_week = 1
    weekly_buckets = defaultdict(list)

    for r in results:
        r.week_after_training = math.ceil((r.seq_date - min_date).days / 7)
        weekly_buckets[r.week_after_training].append(r)
        max_week = max(max_week, r.week_after_training)

    if max_week_spec:
        max_week = max_week_spec

    cumulative_case_counts = 0
    cumulative_seq_counts = 0
    cumulative_lineage_counts = 0
    cumulative_salient_seq_counts = 0
    dated_results = []

    lineages_so_far = set()

    for w in range(1, max_week + 1):
        weekly_data = weekly_buckets[w]
        if weekly_data:
            cumulative_case_counts += sum(x.count for x in weekly_data)
            cumulative_seq_counts += len(weekly_data)
            cumulative_salient_seq_counts += sum(
                1 if x.count >= 10 else 0 for x in weekly_data
            )
            for item in weekly_data:
                if item.new_seq and item.new_pango_flag:
                    if all(l not in lineages_so_far for l in item.lineage):
                        cumulative_lineage_counts += 1

                    for l in item.lineage:
                        lineages_so_far.add(l)

        dated_results.append(
            DatedStats(
                case_counts=cumulative_case_counts,
                seq_counts=cumulative_seq_counts,
                lineage_counts=cumulative_lineage_counts,
                salient_seq_counts=cumulative_salient_seq_counts,
                salient_to_seq_ratio=cumulative_salient_seq_counts / cumulative_seq_counts if cumulative_seq_counts > 0 else 0,
            )
        )

    return dated_results


def get_local_stats_dated_non_cumulative(
    results: List[ScoredData],
    date_cutoff: str,
    max_period_spec: int = None,
    period_size: int = 7,
) -> List[LocalStats]:
    """
    For a date-sorted result set, find out the relevant stats over time
    """
    min_date = datetime.datetime.strptime(date_cutoff, "%Y-%m-%d")
    max_period = 1
    periodly_buckets = defaultdict(list)

    for r in results:
        r.period_after_training = math.ceil((r.seq_date - min_date).days / period_size)
        periodly_buckets[r.period_after_training].append(r)
        max_period = max(max_period, r.period_after_training)

    if max_period_spec:
        max_period = max_period_spec

    # cumulative_case_counts = 0
    # cumulative_seq_counts = 0
    # cumulative_lineage_counts = 0
    # cumulative_salient_seq_counts = 0
    dated_results = []

    lineages_so_far = set()

    for w in range(1, max_period + 1):
        periodly_data = periodly_buckets[w]
        case_counts = 0
        seq_counts = 0
        salient_seq_counts = 0
        lineage_counts = 0

        if periodly_data:
            case_counts = sum(x.count for x in periodly_data)
            seq_counts = len(periodly_data)
            salient_seq_counts = sum(
                1 if x.count >= 10 else 0 for x in periodly_data
            )
            lineage_counts = 0
            for item in periodly_data:
                if item.new_seq and item.new_pango_flag:
                    if all(l not in lineages_so_far for l in item.lineage):
                        lineage_counts += 1

                    for l in item.lineage:
                        lineages_so_far.add(l)

        dated_results.append(
            DatedStats(
                case_counts=case_counts,
                seq_counts=seq_counts,
                lineage_counts=lineage_counts,
                salient_seq_counts=salient_seq_counts,
                salient_to_seq_ratio=salient_seq_counts / seq_counts if seq_counts > 0 else 0,
            )
        )

    return dated_results


def consolidate_stats(stats: list):
    """
    Consolidate quantities across multiple experiments into
    one mean, variance value. Items in stats represent
    consolidated values for different quantities.
    """
    keys = stats[0]._asdict().keys()
    results = {}
    for key in keys:
        values = [getattr(i, key) for i in stats]
        results[key] = [np.mean(values), get95pct_err(values)]
    return results


def consolidate_stats_list(stats: List[list]):
    """
    Consolidate stats when each item is a list of experiments. So, here
    stats are represented as:
        [
            [exp_00, exp_01, ... exp_0n],  # Lane 0
            [exp_10, exp_11, ... exp_1n],  # Lane 1   
        ]
    We want to consolidate across index i for exp_ij for each value of j
    """
    min_length = min(len(s) for s in stats)

    return [
        consolidate_stats(
            [stats[i][j] for i in range(len(stats))]
        ) for j in range(min_length)
    ]


def sort_tools(tool_list: list) -> list:
    results = []

    if "PandoGen" in tool_list:
        results.append("PandoGen")
        tool_list.remove("PandoGen")

    results.extend(sorted(tool_list))

    return results


def local_stats_plotter(
    plot_results,
    metric,
    x_axis: str,
    y_axis: str,
    filename: str,
    plot_ticks: list,
    marker_append: str = "-",
    plt_legend: bool = True,
    ax = None,
    markersize=10,
    legend_vertical: float = 1.0,
) -> None:
    """
    Plot local stats
    """
    if ax is None:
        plt.figure(figsize=(20, 10))
        _, ax = plt.subplots()

    top_p_fmt = {
        "0.95": "+",
        "0.97": "x",
        "0.99": "^",
        "0.995": "s",
        "0.997": "D",
        "1.0": "o",
        "T0": "*",
        "T1": "h",
    }

    tools_encountered = set()
    top_p_encountered = set()
    
    for tool in plot_results:
        for top_p in plot_results[tool]:
            stats_sequence = plot_results[tool][top_p]
            length = len(stats_sequence)
            plot_ticks_ = [p for p in plot_ticks if p < length]
            ticks_sequence = [stats_sequence[p - 1][metric] for p in plot_ticks_]

            if not ticks_sequence:
                logger.info(f"No plot points found for {tool}/{top_p}. Total number of items = {len(stats_sequence)}")
                continue

            mean, err = tuple(zip(*ticks_sequence))
            ax.errorbar(
                plot_ticks_,
                mean,
                fmt=tool_colors[tool] + top_p_fmt[top_p] + marker_append,
                yerr=err,
                alpha=0.5,
                capsize=3,
                label=tool,
                markersize=markersize,
            )
            top_p_encountered.add(top_p)
        tools_encountered.add(tool)
            
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    def get_tool_label(name: str):
        return r"\textbf{" + name + "}" if name == "PandoGen" else name

    tools_encountered = sort_tools(tools_encountered)

    if plt_legend:
        legend_elements = [
            Line2D([0], [0], color=tool_colors[tool], label=get_tool_label(tool), lw=2) for tool in tools_encountered
        ] + [
            Line2D([0], [0], marker=top_p_fmt[top_p][0], color='black', label=str(top_p), markerfacecolor='black', markersize=5)
            for top_p in sorted(top_p_encountered)
        ]
    
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, legend_vertical), loc="upper left")
        plt.savefig(filename, dpi=300, bbox_inches="tight")


def find_key(results_dict: dict, prefix: str) -> str:
    prefix = os.path.abspath(prefix) + "/"
    matching = [key for key in results_dict if key.startswith(prefix)]
    return matching


def main(args):
    if os.path.exists(args.output_dir):
        raise ValueError(f"Directory {args.output_dir} already exists!")

    os.makedirs(args.output_dir)

    if not (args.local_plots or args.global_plots):
        raise ValueError("One of --global_plots or --local_plots should be set")

    """
    JSON file configuration:

    Tool0 ->
        top_p0: directory
        top_p1: directory
        ...
    """
    with open(args.results_config, "r") as fhandle:
        config = json.load(fhandle)

    logger.info(f"Got JSON config,\n{json.dumps(config, indent=4)}")

    """
    results configuration:
    filename0: results_list
    filename1: results_list
    . . .
    filenamen: results_list
    """
    logger.info("Reading and uniquifying results")

    with open(args.results, "rb") as fhandle:
        results = pickle.load(fhandle)

    """
    Reorganize this as
    Tool0 -> 
        top_p0: [
            results_list0,
            results_list1,
            ...
        ],
        top_p1: [
            results_list0,
            ...
        ]
    Tool1 ->
        ...

    Create two other versions where we have stats as well. See code below.
    """
    results = {os.path.abspath(key): value for key, value in results.items()}

    if not args.no_uniq:
        logger.info("Uniquifying data")
        results = {key: uniquify(value) for key, value in results.items()}

    logger.info("Organizing statistics")

    organized_results = defaultdict(lambda: defaultdict(list))
    organized_results_global_stats = defaultdict(lambda: defaultdict(list))
    organized_results_global_stats_reduced = defaultdict(lambda: dict())
    organized_results_local_stats = defaultdict(lambda: defaultdict(list))
    organized_results_local_stats_reduced = defaultdict(lambda: dict())
    organized_results_local_stats_all = defaultdict(lambda: defaultdict(list))
    organized_results_local_stats_all_reduced = defaultdict(lambda: dict())
    organized_results_local_stats_date_sorted = defaultdict(lambda: defaultdict(list))
    organized_results_local_stats_date_sorted_reduced = defaultdict(lambda: dict())
    organized_results_local_stats_date_sorted_non_cumulative = defaultdict(lambda: defaultdict(list))
    organized_results_local_stats_date_sorted_non_cumulative_reduced = defaultdict(lambda: dict())

    for tool in config:
        for top_p, directory in config[tool].items():
            logger.info(f"Processing {tool}, {top_p}, {directory}")
            for f in find_key(results, directory):
                result_for_filename = sorted(results[f], key=lambda x: x.score, reverse=True)
                stats_for_filename = get_global_stats(result_for_filename)
                organized_results[tool][top_p].append(result_for_filename)
                organized_results_global_stats[tool][top_p].append(stats_for_filename)
                organized_results_local_stats[tool][top_p].append(
                    get_local_stats(result_for_filename, local_stat_type=LocalStatType.NOVEL)
                )
                organized_results_local_stats_all[tool][top_p].append(
                    get_local_stats(result_for_filename, local_stat_type=LocalStatType.ALL)
                )

                if args.date_cutoff:
                    result_for_filename_date_sorted = sorted(
                        filter(lambda x: x.new_seq and not x.invalid_seq, results[f]),
                        key=lambda x: x.seq_date
                    )
                    organized_results_local_stats_date_sorted[tool][top_p].append(
                        get_local_stats_dated(result_for_filename_date_sorted, args.date_cutoff, max_week_spec=75)
                    )
                    organized_results_local_stats_date_sorted_non_cumulative[tool][top_p].append(
                        get_local_stats_dated_non_cumulative(
                            result_for_filename_date_sorted,
                            args.date_cutoff,
                            max_period_spec=18,
                            period_size=28,
                        )
                    )

            organized_results_global_stats_reduced[tool][top_p] = consolidate_stats(
                organized_results_global_stats[tool][top_p]
            )
            
            organized_results_local_stats_reduced[tool][top_p] = consolidate_stats_list(
                organized_results_local_stats[tool][top_p]
            )

            organized_results_local_stats_all_reduced[tool][top_p] = consolidate_stats_list(
                organized_results_local_stats_all[tool][top_p]
            )

            if args.date_cutoff:
                organized_results_local_stats_date_sorted_reduced[tool][top_p] = consolidate_stats_list(
                    organized_results_local_stats_date_sorted[tool][top_p]
                )
                organized_results_local_stats_date_sorted_non_cumulative_reduced[tool][top_p] = consolidate_stats_list(
                    organized_results_local_stats_date_sorted_non_cumulative[tool][top_p]
                )

    logger.info(f"Global stats:\n{json.dumps(organized_results_global_stats_reduced, indent=4)}")

    """
    Plot global stats
    """
    if args.global_plots:
        logger.info("Plotting global stats")

        global_stats_plotter(
            organized_results_global_stats_reduced,
            args.output_dir,
            x_axis_key=args.global_x_axis_key,
            x_axis_name=args.global_x_axis_name,
            min_non_old=args.min_novel_seq,
        )

    """
    Local stats plotter
    """
    if args.local_plots:
        logger.info("Plotting local stats")

        local_stats_plotter(
            plot_results=organized_results_local_stats_reduced,
            metric="rate",
            x_axis="Novel sequence rank",
            y_axis="PPV",
            filename=os.path.join(args.output_dir, "PPV.png"),
            plot_ticks=list(range(51, 1001, 50)),
        )

        local_stats_plotter(
            plot_results=organized_results_local_stats_reduced,
            metric="tp_count",
            x_axis="Novel sequence rank",
            y_axis="Case counts",
            filename=os.path.join(args.output_dir, "CaseCounts.png"),
            plot_ticks=list(range(51, 1001, 50)),
        )

        local_stats_plotter(
            plot_results=organized_results_local_stats_all_reduced,
            metric="rate",
            x_axis="Sequence rank",
            y_axis="Efficiency",
            filename=os.path.join(args.output_dir, "Efficiency.png"),
            plot_ticks=list(range(51, 1001, 50)),
        )

    """
    Plot local stats for dated stats
    """
    if args.local_plots and args.date_cutoff:
        matplotlib.rc('font', size=40)
        logger.info("Plotting local dated stats")
        plt.figure(figsize=(20, 20))
        ax = plt.subplot(211)
        local_stats_plotter(
            plot_results=organized_results_local_stats_date_sorted_reduced,
            metric="seq_counts",
            x_axis="Weeks after training",
            y_axis=r"\#Forecasts",
            filename=None,
            plot_ticks=list(range(1, 85, 5)),
            plt_legend=False,
            ax=ax,
            markersize=20,
            legend_vertical=1.5,
        )

        ax = plt.subplot(212)
        local_stats_plotter(
            plot_results=organized_results_local_stats_date_sorted_reduced,
            metric="salient_seq_counts",
            x_axis="Weeks after training",
            y_axis=r"\#Salient Forecasts",
            filename=os.path.join(args.output_dir, "StatsOverTime.png"),
            plot_ticks=list(range(1, 85, 5)),
            plt_legend=True,
            ax=ax,
            markersize=20,
            legend_vertical=1.5,
        )
        

        plt.figure(figsize=(20, 20))
        ax = plt.subplot(211)
        local_stats_plotter(
            plot_results=organized_results_local_stats_date_sorted_non_cumulative_reduced,
            metric="seq_counts",
            x_axis="Months after training",
            y_axis=r"\#Forecasts",
            filename=None,
            plot_ticks=list(range(1, 16)),
            plt_legend=False,
            ax=ax,
            markersize=20,
            legend_vertical=1.5,
        )

        ax = plt.subplot(212)
        local_stats_plotter(
            plot_results=organized_results_local_stats_date_sorted_non_cumulative_reduced,
            metric="salient_seq_counts",
            x_axis="Months after training",
            y_axis=r"\#Salient Forecasts",
            filename=os.path.join(args.output_dir, "StatsOverTimeNonCumulative.png"),
            plot_ticks=list(range(1, 16)),
            plt_legend=True,
            ax=ax,
            markersize=20,
            legend_vertical=1.5,
        )

    
if __name__ == "__main__":
    parser = ArgumentParser(description="Plot results")

    parser.add_argument(
        "--results_config",
        help="JSON mapping output directory to tool and top_p",
        required=True,
    )

    parser.add_argument(
        "--results",
        help="Eval results pickle file",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        help="Output directory (should not exist already)",
        required=True,
    )

    parser.add_argument(
        "--global_plots",
        help="Do global plots",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--local_plots",
        help="Do local plots",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--global_x_axis_key",
        help="Global X-axis key",
        default=None,
    )

    parser.add_argument(
        "--global_x_axis_name",
        help="X-axis name to be used",
        default=None,
    )

    parser.add_argument(
        "--no_uniq",
        default=False,
        action="store_true",
        help="Do not uniquify data (for testing)",
    )

    parser.add_argument(
        "--min_novel_seq",
        required=False,
        help="The minimum number of novel sequences for an operating point to be included",
        type=int,
    )

    parser.add_argument(
        "--date_cutoff",
        help="Cut-off date for training to calculate datewise stats (if not provided, dated stats won't be plotted)",
        required=False,
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    main(args)
