# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pickle
from collections import namedtuple
import matplotlib
from matplotlib import pyplot as plt
import json
import sys
from find_distance import ScoredData
import os
from typing import List, Optional, Callable
matplotlib.rcParams.update({'font.size': 22})
import scipy.stats
from utils import ambiguity_mapping
import numpy as np
from dataclasses import dataclass
from matplotlib.lines import Line2D
import re
import copy
from collections import defaultdict
import argparse
import logging
import hashlib

logger = logging.getLogger(__file__)

_TICKS = ["p0.95", "p0.97", "p0.99", "p0.995", "p0.997", "p1.0"]

GlobalStats = namedtuple(
    "GlobalStats", [
        "n_valid",
        "n_new",
        "n_false",
        "n_old",
        "avg_kmer_novelty",
        "old_rate",
        "new_rate",
        "ppv",
        "counts",
        "n_non_old",
        "n_new_lineages",
        "case_count_per_forecast",
        "case_count_per_novel",
        "n_salient_sequences",
    ]
)


@dataclass
class ResultsDictionary:
    """
    :param res_key_mapping_json: Filename of json configuration
    :param res_key_mapping: key in eval pickle file mapped to tool name and top_p config
    :param res_keys: eval pickle key prefixes mapped to empty lists
    :param keys_consolidated: keys at the tool-level
    """
    res_key_mapping_json: str
    res_key_mapping: Optional[dict] = None
    res_keys: Optional[dict] = None
    keys_consolidated: Optional[dict] = None

    def __post_init__(self):
        if not self.res_key_mapping:
            with open(self.res_key_mapping_json) as fhandle:
                self.res_key_mapping = json.load(fhandle)

        if not self.res_keys:
            self.res_keys = {key: [] for key in self.res_key_mapping}

        if not self.keys_consolidated:
            self.keys_consolidated = dict()

            for key, value in self.res_key_mapping.items():
                self.keys_consolidated[key] = re.sub(r"\s*\(.*$", "", value)


def get95pct_err(values: list):
    if len(values) == 0:
        return 0
    
    scale = scipy.stats.sem(values)
    if scale != 0:
        x, y = scipy.stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=scale)
        return (y - x) / 2
    else:
        return 0


def get_results(unique_res):
    tp = []
    fp = []
    metric = []
    tp_counts = []

    for item in unique_res:
        last_tp = tp[-1] if tp else 0
        last_fp = fp[-1] if fp else 0
        last_tp_count = tp_counts[-1] if tp_counts else 0
        
        if item.new_seq:
            tp.append(last_tp + 1)
            tp_counts.append(last_tp_count + item.count)
            fp.append(last_fp)
            metric.append(item.score)
        elif item.false_seq:
            fp.append(last_fp + 1)
            tp.append(last_tp)
            metric.append(item.score)
            tp_counts.append(last_tp_count)
            
    return fp, tp, tp_counts


def get_results_all_sequences(unique_res):
    """
    Compare new novel sequences to all sequence generations
    """
    tp = []
    fp = []
    metric = []
    tp_counts = []

    for item in unique_res:
        last_tp = tp[-1] if tp else 0
        last_fp = fp[-1] if fp else 0
        last_tp_count = tp_counts[-1] if tp_counts else 0
        
        if item.new_seq:
            tp.append(last_tp + 1)
            tp_counts.append(last_tp_count + item.count)
            fp.append(last_fp)
            metric.append(item.score)
        else:
            fp.append(last_fp + 1)
            tp.append(last_tp)
            metric.append(item.score)
            tp_counts.append(last_tp_count)
            
    return fp, tp, tp_counts


def eq(a, b):
    result = False
    
    if len(a) != len(b):
        return result
    
    for i, j in zip(a, b):
        if i == j or "X" in [i, j]:
            continue
            
        if i in ambiguity_mapping.get(j, []):
            continue
            
        if j in ambiguity_mapping.get(i, []):
            continue
            
        break
    else:
        result = True
        
    return result


def uniquify(l: list) -> list:
    unique_set = set()
    res = []
    
    for i in range(len(l)):
        a = l[i]

        if i == 0:
            unique_set.add(a.seq)
            res.append(a)
        else:
            eq_flag = a.seq in unique_set or any(eq(a.seq, l[j].seq) for j in range(i))
            if not eq_flag:
                unique_set.add(a.seq)
                res.append(a)
            
    return res


def get_ranges(points: list, results: list):
    collected_ranges = {p: [] for p in points}
    for p in points:
        for r in results:
            fp, tp, tp_count = r
            try:
                collected_ranges[p].append(
                    [
                        tp[p - 1] / (tp[p - 1] + fp[p - 1]),
                        tp_count[p - 1]
                    ]
                )
            except IndexError as e:
                break

    return collected_ranges


def get_global_stats(items: List[ScoredData]) -> tuple:
    total_valid_sequences = 0
    total_new_sequences = 0
    total_false_sequences = 0
    total_old_sequences = 0
    total_kmer_novelty = 0
    total_case_counts = 0
    total_non_old = 0
    
    seq_set = set()
    
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
    )


def get_stats_per_prefix(
    results: dict,
    groupings: dict,
    functor: Callable = get_global_stats
):
    """
    Collect results for each method for each top_p value together. For example,
    all PandoGen experiments under top_p = 0.95 will be collected into one list,
    indexed by PandoGen and the top_p value. The exact key to use is provided
    under "groupings"
    """
    for key in results:
        try:
            grouping_key = [g for g in groupings if key.startswith(g + "/")].pop()
        except Exception as e:
            logger.info(f"Key {key} doesn't have a grouping. Skipping.")
            continue

        groupings[grouping_key].append(functor(results[key]))
                
    return groupings


def consolidate_global_stats(stats: list):
    """
    Each item in GlobalStats object is converted to a mean
    and error value
    """
    keys = stats[0]._asdict().keys()
    results = {}
    for key in keys:
        values = [getattr(i, key) for i in stats]
        results[key] = [np.mean(values), get95pct_err(values)]
    return results


def consolidate_results(res: dict, res_spec: ResultsDictionary):
    # 1. Collect results under each tool and each top_p value as a list.
    # Results are items in the GlobalStats object, keys are the same as that
    # in res_spec
    global_stats = get_stats_per_prefix(res, copy.deepcopy(res_spec.res_keys))

    # 2. Convert each list into a single mean/error value. There are two levels
    # of keys here: key1->key2->(mean, err), where key1 is from res_spec.res_keys
    # and key2 is from GlobalStats
    consolidated_global_stats = {
        key: consolidate_global_stats(value) for key, value in global_stats.items()}

    logger.info(f"Global stats:\n{json.dumps(consolidated_global_stats)}")

    return consolidated_global_stats


def get_global_stat_plots(res: dict, res_spec: ResultsDictionary, output_dir: str):
    consolidated_global_stats = consolidate_results(res, res_spec)

    ticks = _TICKS
    nticks = [float(x[1:]) for x in ticks]

    # Here we create one more level
    # key1->key2->key3, where key1 is tool name, key2 is p0.95 etc, and key3 is the metric
    consolidated_tool_level_global_stats = defaultdict(dict)

    for key in consolidated_global_stats:
        mapping_tool = res_spec.keys_consolidated[key]
        p_tick = re.findall(r"p[0-9]\.[0-9]+", key).pop()
        logger.info(f"Assigning {mapping_tool} -> {p_tick}, orig_key = {key}")
        consolidated_tool_level_global_stats[mapping_tool][p_tick] = consolidated_global_stats[key]

    print(json.dumps(consolidated_tool_level_global_stats, indent=4))

    def plot_helper(method, key, ax, fmt, y_name):
        mean, yerr = [], []

        for ptick in ticks:
            try:
                mean.append(consolidated_tool_level_global_stats[method][ptick][key][0])
                yerr.append(consolidated_tool_level_global_stats[method][ptick][key][1])
            except KeyError as e:
                logger.error(f"Cannot find something: {method}, {key}, {ptick}")
                raise e

        ax.errorbar(
                nticks,
                mean,
                fmt=fmt,
                yerr=yerr,
                alpha=0.5,
                capsize=3,
                label=key,
        )
        ax.set_ylabel(y_name)

    fmt_keys = {
        "PandoGen": "ro-",
        "Prot GPT2 unenumerated": "yx-",
        "Prot GPT2 enumerated": "ys-",
        "SDA": "b^-",
    }

    plt.figure(figsize=(20, 7))
    ax = plt.subplot(221)

    for key in res_spec.keys_consolidated.values():
        plot_helper(key, "ppv", ax, fmt=fmt_keys[key], y_name="PPV")

    # plot_helper("PandoGen", "ppv", ax, fmt="ro-", y_name="PPV")
    # plot_helper("Prot GPT2 unenumerated", "ppv", ax, fmt="yx-", y_name="PPV")
    # plot_helper("Prot GPT2 enumerated", "ppv", ax, fmt="ys-", y_name="PPV")
    # plot_helper("SDA", "ppv", ax, fmt="b^-", y_name="PPV")

    ax = plt.subplot(222)

    for key in res_spec.keys_consolidated.values():
        plot_helper(key, "counts", ax, fmt=fmt_keys[key], y_name="Case counts")

    # plot_helper("PandoGen", "counts", ax, fmt="ro-", y_name="Case counts")
    # plot_helper("Prot GPT2 unenumerated", "counts", ax, fmt="yx-", y_name="Case counts")
    # plot_helper("Prot GPT2 enumerated", "counts", ax, fmt="ys-", y_name="Case counts")
    # plot_helper("SDA", "counts", ax, fmt="b^-", y_name="Case counts")

    ax = plt.subplot(223)

    for key in res_spec.keys_consolidated.values():
        plot_helper(key, "n_new", ax, fmt=fmt_keys[key], y_name="#new_sequences")

    # plot_helper("PandoGen", "n_new", ax, fmt="ro-", y_name="#new sequences")
    # plot_helper("Prot GPT2 unenumerated", "n_new", ax, fmt="yx-", y_name="#new sequences")
    # plot_helper("Prot GPT2 enumerated", "n_new", ax, fmt="ys-", y_name="#new sequences")
    # plot_helper("SDA", "n_new", ax, fmt="b^-", y_name="#new sequences")

    ax = plt.subplot(224)

    for key in res_spec.keys_consolidated.values():
        plot_helper(key, "avg_kmer_novelty", ax, fmt=fmt_keys[key], y_name="Sample distance")

    # plot_helper("PandoGen", "avg_kmer_novelty", ax, fmt="ro-", y_name="Sample distance")
    # plot_helper("Prot GPT2 unenumerated", "avg_kmer_novelty", ax, fmt="yx-", y_name="Sample difference")
    # plot_helper("Prot GPT2 enumerated", "avg_kmer_novelty", ax, fmt="ys-", y_name="Sample difference")
    # plot_helper("SDA", "avg_kmer_novelty", ax, fmt="b^-", y_name="Sample distance")

    legend_dict = {
        "PandoGen": Line2D([0], [0], marker='o', color='r', label='PandoGen', markerfacecolor='red', markersize=5),
        "Prot GPT2 unenumerated": Line2D([0], [0], marker='x', color='y', label='Prot GPT2 unenumerated', markerfacecolor='yellow', markersize=5),
        "Prot GPT2 enumerated": Line2D([0], [0], marker='s', color='y', label='Prot GPT2 enumerated', markerfacecolor='yellow', markersize=5),
        "SDA": Line2D([0], [0], marker='^', color='b', label='SDA', markerfacecolor='b', markersize=5),
    }

    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='r', label='PandoGen', markerfacecolor='red', markersize=5),
    #     Line2D([0], [0], marker='x', color='y', label='Prot GPT2 unenumerated', markerfacecolor='yellow', markersize=5),
    #     Line2D([0], [0], marker='s', color='y', label='Prot GPT2 enumerated', markerfacecolor='yellow', markersize=5),
    #     Line2D([0], [0], marker='^', color='b', label='SDA', markerfacecolor='b', markersize=5),
    # ]
    legend_elements = [
        legend_dict[key] for key in set(res_spec.keys_consolidated.values())
    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig(
        os.path.join(output_dir, "Macro.png"),
        dpi=300,
        bbox_inches="tight",
    )

    max_counts = {}

    for tool in consolidated_tool_level_global_stats:
        max_counts[tool] = max(
            x["counts"][0] for x in consolidated_tool_level_global_stats[tool].values())

    print("Case counts from novel sequences")
    print("================================")
    print(json.dumps(max_counts, indent=4))

    max_n_new = {}

    for tool in consolidated_tool_level_global_stats:
        max_n_new[tool] = max(
            x["n_new"][0] for x in consolidated_tool_level_global_stats[tool].values())

    print("Maximum number of novel sequences")
    print("=================================")
    print(json.dumps(max_n_new, indent=4))


def prepare_plot_data_individual(results_dictionary: dict, key_to_plot_map: dict, plot_ticks: list):
    results = {}
    
    for key in results_dictionary:
        results_for_key = get_ranges(plot_ticks, results_dictionary[key])
        method_name_for_key = key_to_plot_map[key]
        results[method_name_for_key] = {
            "Yields": {"mean": [], "std": []},
            "Case count": {"mean": [], "std": []}
        }
        
        for ptick in plot_ticks:
            results_for_key_for_tick = results_for_key[ptick]
            
            if not results_for_key_for_tick:
                continue
            
            yield_frac, case_counts = tuple(zip(*results_for_key_for_tick))
            mean_yield = np.mean(yield_frac)
            mean_case_count = np.mean(case_counts)
            err_yield = get95pct_err(yield_frac)
            err_case_count = get95pct_err(case_counts)  # np.std(case_counts)
            results[method_name_for_key]["Yields"]["mean"].append(mean_yield)
            results[method_name_for_key]["Yields"]["std"].append(err_yield)
            results[method_name_for_key]["Case count"]["mean"].append(mean_case_count)
            results[method_name_for_key]["Case count"]["std"].append(err_case_count)
            
    return results


def plot_helper(
    plot_results,
    metric,
    x_axis: str,
    y_axis: str,
    filename: str,
    plot_ticks: list,
    res_spec: ResultsDictionary,
) -> None:
    matplotlib.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots()
    
    colors = {
        "PandoGen (p0.95)": "g+-",
        "PandoGen (p0.97)": "gx-",
        "PandoGen (p0.99)": "g^-",
        "PandoGen (p0.995)": "gs-",
        "PandoGen (p0.997)": "gD-",
        "PandoGen (p1.0)": "go-",
        
        "Prot GPT2 enumerated (p0.95)": "b+-",
        "Prot GPT2 enumerated (p0.97)": "bx-",
        "Prot GPT2 enumerated (p0.99)": "b^-",
        "Prot GPT2 enumerated (p0.995)": "bs-",
        "Prot GPT2 enumerated (p0.997)": "bD-",
        "Prot GPT2 enumerated (p1.0)": "bo-",
        "Prot GPT2 unenumerated (p0.95)": "r+-",
        "Prot GPT2 unenumerated (p0.97)": "rx-",
        "Prot GPT2 unenumerated (p0.99)": "r^-",
        "Prot GPT2 unenumerated (p0.995)": "rs-",
        "Prot GPT2 unenumerated (p0.997)": "rD-",
        "Prot GPT2 unenumerated (p1.0)": "ro-",
        
        "SDA (p0.95)": "y+-",
        "SDA (p0.97)": "yx-",
        "SDA (p0.99)": "y^-",
        "SDA (p0.995)": "ys-",
        "SDA (p0.997)": "yD-",
        "SDA (p1.0)": "yo-",
    }

    for i, key in enumerate(plot_results):
        c = colors[key]
        length = len(plot_results[key][metric]["mean"])
        assert(length == len(plot_results[key][metric]["std"]))
        ax.errorbar(
            plot_ticks[:length],
            plot_results[key][metric]["mean"],
            fmt=colors[key],
            yerr=plot_results[key][metric]["std"],
            alpha=0.5,
            # ecolor='black',
            capsize=3,
            label=key,
        )
    
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    legend_dict = {
        'Prot GPT2 unenumerated': Line2D([0], [0], color='red', label='Prot GPT2 unenumerated', lw=2),
        'Prot GPT2 enumerated': Line2D([0], [0], color='blue', label='Prot GPT2 enumerated', lw=2),
        'PandoGen': Line2D([0], [0], color='green', label='PandoGen', lw=2),
        'SDA': Line2D([0], [0], color='yellow', label='SDA', lw=2),
    }
    
    legend_elements = [
        legend_dict[k] for k in set(res_spec.keys_consolidated.values())
    ] + [
        # Line2D([0], [0], color='red', label='Prot GPT2 unenumerated', lw=2),
        # Line2D([0], [0], color='blue', label='Prot GPT2 enumerated', lw=2),
        # Line2D([0], [0], color='green', label='PandoGen', lw=2),
        # Line2D([0], [0], color='yellow', label='SDA', lw=2),
        Line2D([0], [0], marker='+', color='black', label='p=0.95', markerfacecolor='black', markersize=5),
        Line2D([0], [0], marker='x', color='black', label='p=0.97', markerfacecolor='black', markersize=5),
        Line2D([0], [0], marker='^', color='black', label='p=0.99', markerfacecolor='black', markersize=5),
        Line2D([0], [0], marker='s', color='black', label='p=0.995', markerfacecolor='black', markersize=5),
        Line2D([0], [0], marker='D', color='black', label='p=0.997', markerfacecolor='black', markersize=5),
        Line2D([0], [0], marker='o', color='black', label='p=1.00', markerfacecolor='black', markersize=5),
    ]
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def temperature_scaling_macro_plotter(res: dict, res_spec: ResultsDictionary, output_dir: str):
    consolidated_global_stats = consolidate_results(res, res_spec)

    # Group so that we get per-tool, stats for each kmer-novelty value
    grouped_results = dict()

    for key, value in consolidated_global_stats.items():
        kmer_novelty = value["avg_kmer_novelty"]
        key0 = res_spec.keys_consolidated[key]
        key1 = tuple(kmer_novelty)
        if key0 not in grouped_results:
            grouped_results[key0] = {}
        grouped_results[key0][key1] = value

    def plot_helper(method, key, ax, fmt, y_name):
        tool = method

        x_axis = []
        x_err = []
        y_axis = []
        y_err = []

        for mean, std in grouped_results[tool]:
            results_operating_point = grouped_results[tool][(mean, std)]
            result_subtype_mean, result_subtype_std = results_operating_point[key]
            x_axis.append(mean)
            x_err.append(std)
            y_axis.append(result_subtype_mean)
            y_err.append(result_subtype_std)

        ax.errorbar(
            x_axis,
            y_axis,
            fmt=fmt,
            xerr=x_err,
            yerr=y_err,
            alpha=0.5,
            capsize=3,
            label=key,
        )
        ax.set_ylabel(y_name)
        ax.set_xlabel("Sample k-mer difference")

    matplotlib.rcParams.update({'font.size': 16})
    plt.figure(figsize=(16, 9))
    ax = plt.subplot(221)
    plot_helper("PandoGen", "ppv", ax, fmt="go", y_name="PPV")
    plot_helper("Prot GPT2 unenumerated", "ppv", ax, fmt="b^", y_name="PPV")
    plot_helper("Prot GPT2 enumerated", "ppv", ax, fmt="yx", y_name="PPV")
    ax = plt.subplot(222)
    plot_helper("PandoGen", "counts", ax, fmt="go", y_name="Case counts")
    plot_helper("Prot GPT2 unenumerated", "counts", ax, fmt="b^", y_name="Case counts")
    plot_helper("Prot GPT2 enumerated", "counts", ax, fmt="yx", y_name="Case counts")
    ax = plt.subplot(223)
    plot_helper("PandoGen", "n_new", ax, fmt="go", y_name="#new sequences")
    plot_helper("Prot GPT2 unenumerated", "n_new", ax, fmt="b^", y_name="#new sequences")
    plot_helper("Prot GPT2 enumerated", "n_new", ax, fmt="yx", y_name="#new sequences")

    legend_elements = [
            Line2D([0], [0], marker='o', color='g', label='PandoGen', markerfacecolor='green', markersize=5),
            Line2D([0], [0], marker='^', color='b', label='Prot GPT2 (unenumerated)', markerfacecolor='blue', markersize=5),
            Line2D([0], [0], marker='x', color='y', label='Prot GPT2 (enumerated)', markerfacecolor='yellow', markersize=5),
    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0), loc="upper left")

    plt.savefig(
        os.path.join(output_dir, "Macro_temperature.png"), dpi=300, bbox_inches="tight")


def main(args):
    if os.path.exists(args.workdir):
        raise ValueError("Provide a non-empty directory path")
    
    res_spec = ResultsDictionary(args.res_spec)

    logger.info("Reading results and uniquifying")
    
    with open(args.eval_filename, "rb") as fhandle:
        res = pickle.load(fhandle)

    res = {key: uniquify(value) for key, value in res.items()}

    logger.info("Plotting global stats")

    if args.temperature_scaling:
        global_temperature_stats = os.path.join(args.workdir, "global_stats_temperature")
        os.makedirs(global_temperature_stats)
        temperature_scaling_macro_plotter(res, res_spec, output_dir=global_temperature_stats)
    else:
        global_stats_workdir = os.path.join(args.workdir, "global_stats")
        local_stats_workdir = os.path.join(args.workdir, "local_stats")
        os.makedirs(global_stats_workdir)
        os.makedirs(local_stats_workdir)

        get_global_stat_plots(res, res_spec, output_dir=global_stats_workdir)

        logger.info("Plotting local stats: PPV, case counts")

        plot_ticks = list(range(50, 1001, 50))

        results_for_plot = prepare_plot_data_individual(
            get_stats_per_prefix(res, copy.deepcopy(res_spec.res_keys), get_results),
            key_to_plot_map=res_spec.res_key_mapping,
            plot_ticks=plot_ticks,
        )

        plot_helper(
            results_for_plot,
            "Yields",
            x_axis="Top-n new sequences",
            y_axis="PPV",
            filename=os.path.join(local_stats_workdir, "PPV.png"),
            plot_ticks=plot_ticks,
            res_spec=res_spec,
        )

        plot_helper(
            results_for_plot,
            "Case count",
            x_axis="Top-n new sequences",
            y_axis="Case counts (GISAD)",
            filename=os.path.join(local_stats_workdir, "CaseCounts.png"),
            plot_ticks=plot_ticks,
            res_spec=res_spec,
        )

        logger.info("Plotting local stats: Yield")

        results_for_plot = prepare_plot_data_individual(
            get_stats_per_prefix(res, copy.deepcopy(res_spec.res_keys), get_results_all_sequences),
            key_to_plot_map=res_spec.res_key_mapping,
            plot_ticks=plot_ticks,
        )

        plot_helper(
            results_for_plot,
            metric="Yields",
            x_axis="Top-n all sequences",
            y_axis="Efficiency: #Real & Novel / #Sequences",
            filename=os.path.join(local_stats_workdir, "Efficiency.png"),
            plot_ticks=plot_ticks,
            res_spec=res_spec,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    parser = argparse.ArgumentParser(description="Prepare plot data for PandoGen")

    parser.add_argument(
        "--workdir",
        help="Output directory path",
        required=True,
    )

    parser.add_argument(
        "--res_spec",
        help="Specification of results",
        required=True,
    )

    parser.add_argument(
        "--eval_filename",
        help="Evaluation results file",
        required=True,
    )

    parser.add_argument(
        "--temperature_scaling",
        action="store_true",
        help="Indicate whether the current evaluation is for temperature scaling",
        default=False,
    )

    args = parser.parse_args()

    main(args)
