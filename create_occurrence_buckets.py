# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
"""
Create occurrence buckets
"""
import pandas_utils
import pandas
import datetime
from typing import Generator, List, Tuple, Callable, Optional, Union
import itertools
from collections import namedtuple
import copy
import random
import logging
from functools import reduce, partial
from operator import concat
import argparse
import json
from collections import namedtuple
import scipy.stats
import tqdm
import numpy as np
from dataclasses import dataclass
import math
import re

logger = logging.getLogger(__file__)

_REGION_STRINGS = ["Europe"]
_GET_WEEKLY_COUNTS = False

"""
A bracket is defined by the ratio of the counts of the low-occurrence sequence
to that of the high-occurrence sequence.

For each bracket, we also define tolerance, which indicates the time difference
between Submission dates that is acceptable. If the time difference is more than that,
we skip the pair. In addition, we have a lead_time variable which indicates the amount of
time for which the sequences must be counted after discovery.

When the ratio is very low, we allow a larger tolerance, but also demand a longer lead time.
"""
CompetitionBrackets = namedtuple("CompetitionBrackets", ["lo", "hi", "tolerance"])

_FRACTIONS = (
    CompetitionBrackets(lo=0, hi=1e-5, tolerance=5),
    CompetitionBrackets(lo=1e-5, hi=1e-4, tolerance=4),
    CompetitionBrackets(lo=1e-4, hi=1e-3, tolerance=3),
    CompetitionBrackets(lo=1e-3, hi=1e-2, tolerance=2),
    CompetitionBrackets(lo=1e-2, hi=1e-1, tolerance=1),
    CompetitionBrackets(lo=1e-1, hi=1, tolerance=0),
)

SeqPair = namedtuple(
    "SeqPair",
    ["seq0", "count0", "seq1", "count1", "period", "weekly_counts0", "weekly_counts1"],
    defaults=(None, None),
)

_MAX_FRAC = max(k.hi for k in _FRACTIONS)
_LEAD_TIME_TO_TOLERANCE_RATIO = 2
_MIN_LEAD_TIME = 8
_MIN_OCCURRENCE_MAX_SEQUENCE = 10

"""
Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5426219/
Especially when more than 20% of cells have expected frequencies < 5, we need to use Fisher's
exact test because applying approximation method is inadequate. 
"""
_MIN_CHI2_SCALAR = 5


@dataclass
class MutationDetails:
    mutation: str
    count: Union[list, np.ndarray]
    week: int
    last_week: int

    def __post_init__(self):
        if isinstance(self.count, np.ndarray):
            return

        count_actual = np.zeros(self.last_week + 1)

        for i in range(len(self.count)):
            count_actual[i + self.week] = self.count[i]

        self.count = np.cumsum(count_actual)

    def get_count_for_week_offset(self, wk: int) -> int:
        return self.count[self.week + wk - 1]

    def get_counts_no_offset(self):
        return self.count[self.week: ].tolist()


def get_slack(lo_frac: float) -> CompetitionBrackets:
    res = [x for x in _FRACTIONS if x.lo <= lo_frac < x.hi]
    if res:
        return res.pop()

    return None


def read_data(
    tsv: str,
    datefield: str = "Submission date",
    protein: str = "Spike",
) -> pandas.DataFrame:
    df = pandas_utils.read_data_frame_cached(tsv, datefield=datefield, protein=protein)
    df = df[df[datefield].str.match(r"^\d+-\d+.*")]
    df = df[df[f"reconstruction_success_{protein}"]]
    return df


def filter_locations(
    df: pandas.DataFrame,
    region_strings: List[str] = _REGION_STRINGS,
    exclude: bool = False,
):
    def location_helper(location: str, r: str):
        return (r not in location) if exclude else (r in location)

    if exclude:
        df = df[df.apply(lambda x: all(location_helper(x.Location, r) for r in region_strings), 1)]
    else:
        df = df[df.apply(lambda x: any(location_helper(x.Location, r) for r in region_strings), 1)]

    return df


def parse_date(d: str) -> datetime.datetime:
    if d is None:
        return None

    return datetime.datetime.strptime(d, "%Y-%m-%d")


def get_counts(df: pandas.DataFrame) -> dict:
    df = df.groupby(["SpikeMutations", "week"], as_index=False)["Accession ID"].count()
    df = df.assign(weekly_counts=df[["week", "Accession ID"]].agg(tuple, axis=1))
    df = df.groupby("SpikeMutations")["weekly_counts"].apply(list)
    return df


def accumulate_weekly_counts(weekly_counts: List[tuple]) -> list:
    min_week = min(x[0] for x in weekly_counts)
    max_week = max(x[0] for x in weekly_counts)
    return_list = [0 for i in range(min_week, max_week + 1)]

    for item in weekly_counts:
        return_list[item[0] - min_week] = item[1]

    return return_list


def get_rare_sequences_with_no_trailing_counts_and_max_count(
    df: pandas.DataFrame,
    trailing_weeks: list,
    max_total: int = 2,
) -> list:
    results = []
    
    for seq, weekly_counts in zip(df.index, df):
        weeks, counts = tuple(zip(*weekly_counts))
        total_counts = sum(counts)
        if len(set(weeks).intersection(set(trailing_weeks))) == 0 and total_counts <= max_total:
            results.append((seq, total_counts))
            
    return results


def get_high_occ_sequences(df: pandas.DataFrame, min_count: int = 500) -> list:
    results = []
    
    for seq, weekly_counts in zip(df.index, df):
        weeks, counts = tuple(zip(*weekly_counts))
        total_counts = sum(counts)
        if total_counts >= min_count:
            results.append((seq, total_counts))
            
    return results


def get_hi_lo_pairings(df: pandas.DataFrame, n_trailing_weeks: int = 4, min_count: int = 500, max_total: int = 2) -> tuple:
    max_week = df.week.max()
    trailing_weeks = list(range(max_week - n_trailing_weeks + 1, max_week + 1))
    df_with_counts = get_counts(df)
    rare_sequences = get_rare_sequences_with_no_trailing_counts_and_max_count(
        df_with_counts, trailing_weeks=trailing_weeks, max_total=max_total
    )
    frequent_sequences = get_high_occ_sequences(df_with_counts, min_count=min_count)
    return frequent_sequences, rare_sequences


def add_hi_lo_pairings(
    train_pairings: List[SeqPair],
    val_pairings: List[SeqPair],
    test_pairings: List[SeqPair],
    frequent_sequences: list,
    rare_sequences: list,
    train_seqs_per_bucket: int,
    val_seqs_per_bucket: int,
    test_seqs_per_bucket: int,
    max_pairings_for_one_seq: int = 5000,
    randshuffle_functor: Callable = random.shuffle,
    randsample_functor: Callable = random.sample,
):
    bucket_size = train_seqs_per_bucket + val_seqs_per_bucket + test_seqs_per_bucket
    prior_train_sequences = set(a.seq0 for a in train_pairings).union(set(
        b.seq1 for b in train_pairings
    ))
    prior_val_sequences = set(a.seq0 for a in val_pairings).union(set(
        b.seq1 for b in val_pairings
    ))
    prior_test_sequences = set(a.seq0 for a in test_pairings).union(set(
        b.seq1 for b in test_pairings
    ))

    train_pairings_dict = {(a.seq0, a.seq1): a for a in train_pairings}
    val_pairings_dict = {(a.seq0, a.seq1): a for a in val_pairings}
    test_pairings_dict = {(a.seq0, a.seq1): a for a in test_pairings}

    sequence_counts = {seq: count for seq, count in frequent_sequences}
    sequence_counts.update({seq: count for seq, count in rare_sequences})

    def get_unassigned_sequences(seq_list: list) -> list:
        assignments = dict()
        unassigned = list()

        for seq, count in seq_list:
            if seq in prior_train_sequences:
                assignments[seq] = "train"
            elif seq in prior_val_sequences:
                assignments[seq] = "val"
            elif seq in prior_test_sequences:
                assignments[seq] = "test"
            else:
                unassigned.append((seq, count))

        return assignments, unassigned

    assigned_freq, unassigned_freq = get_unassigned_sequences(frequent_sequences)
    assigned_rare, unassigned_rare = get_unassigned_sequences(rare_sequences)

    def split_helper(seq_list: list, assignments: dict) -> tuple:
        n_train = max(len(seq_list) * train_seqs_per_bucket // bucket_size, 1)
        n_val = max(len(seq_list) * val_seqs_per_bucket // bucket_size, 1)
        randshuffle_functor(seq_list)
        for seq, count in seq_list[:n_train]:
            assignments[seq] = "train"
        for seq, count in seq_list[n_train: n_train + n_val]:
            assignments[seq] = "val"
        for seq, count in seq_list[n_train + n_val: ]:
            assignments[seq] = "test"

    split_helper(unassigned_freq, assigned_freq)
    split_helper(unassigned_rare, assigned_rare)
    
    def get_split_assignments(assignment_dict: dict) -> tuple:
        train = [x for x, y in assignment_dict.items() if y == "train"]
        val = [x for x, y in assignment_dict.items() if y == "val"]
        test = [x for x, y in assignment_dict.items() if y == "test"]
        return train, val, test

    train_freq, val_freq, test_freq = get_split_assignments(assigned_freq)
    train_rare, val_rare, test_rare = get_split_assignments(assigned_rare)

    def product_helper(freq_set: list, rare_set: list):
        for f in freq_set:
            if len(rare_set) > max_pairings_for_one_seq:
                r_pairings = randsample_functor(rare_set, max_pairings_for_one_seq)
            else:
                r_pairings = rare_set

            for r in r_pairings:
                yield SeqPair(
                    seq0=f,
                    count0=sequence_counts[f],
                    seq1=r,
                    count1=sequence_counts[r],
                    period=None,
                )

    # Override and add new pairings
    for pairing in product_helper(train_freq, train_rare):
        train_pairings_dict[(pairing.seq0, pairing.seq1)] = pairing

    for pairing in product_helper(val_freq, val_rare):
        val_pairings_dict[(pairing.seq0, pairing.seq1)] = pairing

    for pairing in product_helper(test_freq, test_rare):
        test_pairings_dict[(pairing.seq0, pairing.seq1)] = pairing

    return (
        list(train_pairings_dict.values()),
        list(val_pairings_dict.values()),
        list(test_pairings_dict.values())
    )


def get_mutation_data(
    df: pandas.DataFrame,
    availability_last_date: str,
    discovery_last_date: Optional[str] = None,
    period_length: int = 7,
    min_date: Optional[datetime.datetime] = None,
) -> List[MutationDetails]:
    availability_last_date = parse_date(availability_last_date)
    discovery_last_date = parse_date(discovery_last_date)
    last_week = (availability_last_date - min_date).days // period_length

    df2 = df
    df2_counts = get_counts(df2)
    df2_counts_dict = {}
    for spike, counts in zip(df2_counts.index, df2_counts):
        df2_counts_dict[spike] = accumulate_weekly_counts(counts)

    if discovery_last_date:
        df1 = df2[df2.ParsedDate <= discovery_last_date]
    else:
        df1 = df2

    df1_discovery_dates = df1.loc[
        df1.groupby("SpikeMutations").ParsedDate.idxmin()]
    
    df1_discovery_dates_and_counts = []

    for row in df1_discovery_dates.itertuples():
        t = MutationDetails(
            row.SpikeMutations,
            df2_counts_dict[row.SpikeMutations],
            row.week,
            last_week=last_week,
        )
        df1_discovery_dates_and_counts.append(t)

    return df1_discovery_dates_and_counts


def train_val_split(
    discovery_dates_and_counts: list,
    train_seqs_per_bucket: int = 4,
    val_seqs_per_bucket: int = 1,
    test_seqs_per_bucket: int = 0,
    randshuffle_functor: Callable = random.shuffle,
) -> tuple:
    discovery_dates_and_counts = sorted(
        discovery_dates_and_counts, key=lambda x: x.count[-1], reverse=True)
    block_size = train_seqs_per_bucket + val_seqs_per_bucket + test_seqs_per_bucket

    train_sequences = []
    val_sequences = []
    test_sequences = []

    for i in range(0, len(discovery_dates_and_counts), block_size):
        bucket = copy.deepcopy(discovery_dates_and_counts[i: i + block_size])
        randshuffle_functor(bucket)
        if test_seqs_per_bucket > 0:
            train_sequences.extend(bucket[: train_seqs_per_bucket])
            val_sequences.extend(bucket[train_seqs_per_bucket: train_seqs_per_bucket + val_seqs_per_bucket])
            test_sequences.extend(bucket[train_seqs_per_bucket + val_seqs_per_bucket: ])
        else:
            train_sequences.extend(bucket[: -val_seqs_per_bucket])
            val_sequences.extend(bucket[-val_seqs_per_bucket: ])

    return train_sequences, val_sequences, test_sequences


def train_val_split_precalculated(
    discovery_dates_and_counts: List[MutationDetails],
    train_sequences: list,
    val_sequences: list,
    test_sequences: list,
) -> tuple:
    trs = set(train_sequences)
    vas = set(val_sequences)
    tes = set(test_sequences)

    train_sequences = []
    val_sequences = []
    test_sequences = []

    for d in discovery_dates_and_counts:
        if d.mutation in trs:
            train_sequences.append(d)
        elif d.mutation in vas:
            val_sequences.append(d)
        elif d.mutation in tes:
            test_sequences.append(d)

    return train_sequences, val_sequences, test_sequences


def get_weekly_discoveries(
    subset: List[MutationDetails]
) -> List[List[MutationDetails]]:
    max_week = max(s.week for s in subset)
    weekly_sequences = [[] for i in range(max_week + 1)]
    for s in subset:
        weekly_sequences[s.week].append(s)
    return weekly_sequences


def concat_lists(*args):
    return reduce(lambda x, y: concat(x, y), args, [])


def get_counts_for_pair(m0: MutationDetails, m1: MutationDetails) -> Tuple[Tuple[int, int], int]:
    last_week = m0.last_week
    assert(last_week == m1.last_week)
    m0_period = last_week - m0.week + 1
    m1_period = last_week - m1.week + 1
    consensus_period = min(m0_period, m1_period)
    return (
        m0.get_count_for_week_offset(consensus_period),
        m1.get_count_for_week_offset(consensus_period)
    ), consensus_period


def get_pairings(
    discovery_dates_and_counts: list,
    last_week: Optional[int] = None,
    max_sequence_comparisons: Optional[int] = None,
):
    weekly_sequences = get_weekly_discoveries(discovery_dates_and_counts)
    max_frac = _MAX_FRAC

    for i, a_weekly_seq_bucket in tqdm.tqdm(
        enumerate(weekly_sequences), total=len(weekly_sequences), desc="Processing weeks"):
        for j, seq in enumerate(a_weekly_seq_bucket):
            if last_week is None:
                last_week = seq.last_week

            # Seq is the Max sequence in the pair. If it is not
            # at least _MIN_OCCURRENCE count large, we are not using it
            if seq.count[-1] < _MIN_OCCURRENCE_MAX_SEQUENCE:
                continue

            max_slack_for_seq = get_slack(1 / seq.count[-1])

            if max_slack_for_seq is None:
                continue

            max_sequence_range = concat_lists(
                *weekly_sequences[
                    max(0, i - max_slack_for_seq.tolerance):
                    i + max_slack_for_seq.tolerance + 1
                ]
            )

            if max_sequence_comparisons and len(max_sequence_range) > max_sequence_comparisons:
                max_sequence_range = random.sample(max_sequence_range, max_sequence_comparisons)

            for paired_seq in max_sequence_range:
                (seq_count, paired_seq_count), lead_time_for_pair = get_counts_for_pair(
                    seq, paired_seq)

                if paired_seq_count > seq_count:
                    continue

                slack = get_slack(paired_seq_count / seq_count)

                if slack is None:
                    continue

                obtained_tolerance = abs(paired_seq.week - seq.week)

                if slack.tolerance < obtained_tolerance:
                    continue

                lead_time = max(
                    _LEAD_TIME_TO_TOLERANCE_RATIO * obtained_tolerance,
                    _MIN_LEAD_TIME,
                )

                if (last_week - seq.week + 1 < lead_time) or (
                    last_week - paired_seq.week + 1 < lead_time):
                    continue

                if _GET_WEEKLY_COUNTS:
                    yield SeqPair(
                        seq0=seq.mutation,
                        count0=seq_count,
                        seq1=paired_seq.mutation,
                        count1=paired_seq_count,
                        period=lead_time_for_pair,
                        weekly_counts0=seq.get_counts_no_offset(),
                        weekly_counts1=paired_seq.get_counts_no_offset(),
                    )
                else:
                    yield SeqPair(
                        seq0=seq.mutation,
                        count0=seq_count,
                        seq1=paired_seq.mutation,
                        count1=paired_seq_count,
                        period=lead_time_for_pair,
                    )


def create_pairings(discovery_dates_and_counts: list) -> Generator[tuple, None, None]:
    for a, b in itertools.permutations(discovery_dates_and_counts, 2):
        lo_frac = min(a.count / b.count, b.count / a.count)
        slack = get_slack(lo_frac)
        if (slack is not None) and (a.week - slack <= b.week < a.week + slack):
            yield a, b


def to_dict(t: List[MutationDetails]):
    return {m.mutation: m for m in t}


def combine_multiple_data(
    t1: List[MutationDetails],
    t2: List[MutationDetails],
    combine_type: str = "intersection",
) -> List[MutationDetails]:
    t1_dict = to_dict(t1)
    t2_dict = to_dict(t2)

    def helper(a: MutationDetails, b: MutationDetails) -> MutationDetails:
        return MutationDetails(
            a.mutation,
            a.count + b.count,
            min(a.week, b.week),
            a.last_week,
        )

    combined = []

    if combine_type == "intersection":
        combined_keys = set(t1_dict.keys()).intersection(t2_dict.keys())
    else:
        combined_keys = set(t1_dict.keys()).union(t2_dict.keys())

    def impute_helper(tdict: dict, item: MutationDetails, tlist: list) -> None:
        new_item = MutationDetails(
            item.mutation,
            np.zeros_like(item.count),
            item.week,
            item.last_week,
        )
        tdict[item.mutation] = new_item
        tlist.append(new_item)

    for key in combined_keys:
        if key not in t1_dict:
            item = t2_dict[key]
            impute_helper(t1_dict, item, t1)
        elif key not in t2_dict:
            item = t1_dict[key]
            impute_helper(t2_dict, item, t2)

        combined.append(helper(t1_dict[key], t2_dict[key]))

    return combined


def test_contingency_table(table: np.ndarray, use_chi2: bool = False) -> float:
    """
    For Fisher Exact Test, the null hypothesis is that the complete table (src_a, src_b)
    comes from the same distribution. The p_value from the two-sided exact test is the
    probability that a randomly selected table has the same or lower probability than the
    given table. Hence, if p_value provides the percentile rank of the table. We choose the table
    if the percentile rank is at least 0.5 (by default).
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html

    When
    using chi2 test we check whether two sets of values arose by chance. The pvalue returned
    by scipy indicates the probability that they are dependent (null hypothesis)
    (p = 1 implies identical distributions - refer:
    https://github.com/scipy/scipy/blob/v1.10.0/scipy/stats/contingency.py#L298). If the
    condition is not satisfied to use the chi2 test, we use fischer's exact test.
    """
    if use_chi2:
        statistic, pvalue, dof, expected_freq = scipy.stats.chi2_contingency(
            observed=table, correction=True)

        # See note above for reference
        if min(np.min(table.flatten()), np.min(expected_freq.flatten())) >= _MIN_CHI2_SCALAR:
            logger.debug(f"Using chi2 {table}")
            return pvalue

    logger.debug(f"Using Fischer Exact Test {table}")

    statistic, pvalue = scipy.stats.fisher_exact(
        table=table,
        alternative="two-sided",
    )

    return pvalue


def validate_pairings(
    pairings: List[SeqPair],
    src_a: List[MutationDetails],
    src_b: List[MutationDetails],
    exact_test_cutoff: float = 1e-2,
    p_value: Optional[float] = None,
    use_chi2: bool = False,
    max_p_diff: Optional[float] = None,
):
    """
    Ensure that src_a and src_b agree on any given pairing
    """
    a_dict = to_dict(src_a)
    b_dict = to_dict(src_b)

    logger.info("Validating pairing distributions")

    validated_results = []

    if not max_p_diff and not p_value:
        raise ValueError("Provide one of max_p_diff or p_value")

    for pair in tqdm.tqdm(pairings, desc="Validating pairings"):
        counts = (pair.count0, pair.count1)
        a_counts, _ = get_counts_for_pair(a_dict[pair.seq0], a_dict[pair.seq1])
        b_counts, _ = get_counts_for_pair(b_dict[pair.seq0], b_dict[pair.seq1])

        try:
            max_frac = max(y / (x + y + 1e-12) for x, y in [counts, a_counts, b_counts])
            min_frac = min(y / (x + y + 1e-12) for x, y in [counts, a_counts, b_counts])
        except RuntimeWarning as e:
            logger.error(f"Caught runtime warning when processing {counts}, {a_counts}, {b_counts}")
            raise e

        if max_frac > 0.5:
            continue

        if max_p_diff and abs(max_frac - min_frac) > max_p_diff:
            continue

        if counts[1] / sum(counts) >= exact_test_cutoff and p_value:
            pvalue = test_contingency_table(
                np.array([a_counts, b_counts]).T, use_chi2=use_chi2)

            if pvalue < p_value:
                continue

        validated_results.append(pair)

    return validated_results


def prepare_tsv(
    tsv: str,
    availability_last_date: str,
    datefield: str = "Submission date",
    protein: str = "Spike",
    period_length: int = 7,
) -> tuple:
    df = read_data(tsv, datefield=datefield, protein=protein)
    df = df[df.ParsedDate <= parse_date(availability_last_date)]
    min_date = df.ParsedDate.min()
    df = df.assign(week=df.apply(lambda x: (x.ParsedDate - min_date).days // period_length, 1))
    return df, min_date


def get_location_helper(m: str) -> int:
    try:
        return int(re.findall(r"[A-Za-z]+(\d+)[A-Za-z]+", m)[0])
    except IndexError as e:
        logger.error(f"Failed getting location for {m}")
        raise e


def sort_helper(m: list) -> list:
    m = [x for x in m if len(x.strip()) > 0]
    return sorted(m, key=get_location_helper)


def overlap_helper(seq: str, mutations: list) -> bool:
    seq_mutation_positions = set(
        get_location_helper(x) for x in seq.split(",") if len(x.strip()) > 0)
    mutation_positions = set(
        get_location_helper(x) for x in mutations if len(x.strip()) > 0)
    return len(seq_mutation_positions.intersection(mutation_positions)) > 0


def add_fake_sequences(
    sequence_pair_list: List[SeqPair],
    n_random_mutations_per_fake: int,
    n_trials_per_seq: int,
    all_mutations: list,
    all_sequences: set,
    randsample_functor: Callable = random.sample,
) -> None:
    seqs = list(set([s.seq0 for s in sequence_pair_list] + [s.seq1 for s in sequence_pair_list]))

    seq_pair_added = []

    for seq in seqs:
        for i in range(n_trials_per_seq):
            mutations = randsample_functor(all_mutations, n_random_mutations_per_fake)

            if overlap_helper(seq, mutations):
                continue

            mutations = ",".join(sort_helper(seq.split(",") + mutations))

            if mutations in all_sequences:
                continue

            seq_pair_added.append(
                SeqPair(seq0=seq, count0=1, seq1=mutations, count1=0, period=None))

    sequence_pair_list.extend(seq_pair_added)


def add_fake_sequences_top(
    df: pandas.DataFrame,
    sequence_pair_list: List[SeqPair],
    n_random_mutations_for_negatives: int,
    n_negative_trials_per_sequence: int,
    all_sequences: Optional[set] = None,
    all_mutations: Optional[list] = None,
    randsample_functor: Callable = random.sample,
) -> Tuple[set, list]:
    if not all_sequences:
        all_sequences = set(df.SpikeMutations.tolist())

    if not all_mutations:
        all_mutations = set()

        for s in all_sequences:
            all_mutations = all_mutations.union(s.strip().split(","))

        all_mutations = [x for x in all_mutations if len(x) > 0]

    add_fake_sequences(
        sequence_pair_list,
        n_random_mutations_per_fake=n_random_mutations_for_negatives,
        n_trials_per_seq=n_negative_trials_per_sequence,
        all_mutations=all_mutations,
        all_sequences=all_sequences,
        randsample_functor=randsample_functor,
    )
    
    return all_sequences, all_mutations


def get_training_pairs_from_tsv(
    tsv: str,
    availability_last_date: str,
    discovery_last_date: Optional[str] = None,
    primary_locations: list = _REGION_STRINGS,
    train_seqs_per_bucket: int = 4,
    val_seqs_per_bucket: int = 1,
    test_seqs_per_bucket: int = 0,
    datefield: str = "Submission date",
    protein: str = "Spike",
    period_length: int = 7,
    randshuffle_functor: Callable = random.shuffle,
    control_locations: Optional[list] = None,
    exclude_control: bool = False,
    num_randomizations: Optional[int] = None,
    exact_test_cutoff: float = 1e-2,
    p_value: Optional[float] = None,
    use_chi2: bool = False,
    max_p_diff: Optional[float] = None,
    n_trailing_weeks_for_rare: Optional[int] = None,
    min_count_high_freq: Optional[int] = None,
    max_count_rare: Optional[int] = None,
    max_pairings_for_one_seq: Optional[int] = None,
    n_random_mutations_for_negatives: Optional[int] = None,
    n_negative_trials_per_sequence_train: int = 1,
    n_negative_trials_per_sequence_val: int = 1,
    train_sequences_pre: Optional[str] = None,
    val_sequences_pre: Optional[str] = None,
    test_sequences_pre: Optional[str] = None,
    combine_type: str = "intersection",
    max_to_verify: int = 1000000,
    max_sequence_comparisons: Optional[int] = None,
) -> Union[list, tuple]:
    if train_sequences_pre:
        assert(val_sequences_pre is not None)

        with open(train_sequences_pre, "r") as fhandle:
            train_sequences_pre = [l.strip() for l in fhandle]

        with open(val_sequences_pre, "r") as fhandle:
            val_sequences_pre = [l.strip() for l in fhandle]

        if test_sequences_pre:
            with open(test_sequences_pre, "r") as fhandle:
                test_sequences_pre = [l.strip() for l in fhandle]
        else:
            test_sequences_pre = []
    else:
        train_sequences_pre = None
        val_sequences_pre = None
        test_sequences_pre = None

    num_randomizations = 1 if not num_randomizations else num_randomizations

    if train_sequences_pre and num_randomizations > 1:
        raise ValueError("Cannot have num_randomizations > 1 and non-empty train_sequences")

    logger.info("Reading tsv")
    df, min_date = prepare_tsv(
        tsv, availability_last_date=availability_last_date,
            datefield=datefield, protein=protein, period_length=period_length)

    logger.info(f"Getting data for location {primary_locations}")
    df_a = filter_locations(df, region_strings=primary_locations)
    dates_and_counts_a = get_mutation_data(
        df_a,
        availability_last_date,
        discovery_last_date,
        period_length,
        min_date=min_date,
    )
    logger.info(f"Obtained {len(dates_and_counts_a)} items")

    if control_locations:
        prefix = "Getting" if not exclude_control else "Excluding"
        logger.info(f"{prefix} data for location {control_locations}")
        df_b = filter_locations(df, region_strings=control_locations, exclude=exclude_control)
        dates_and_counts_b = get_mutation_data(
            df_b,
            availability_last_date,
            discovery_last_date,
            period_length,
            min_date=min_date,
        )
        logger.info(f"Obtained {len(dates_and_counts_b)} items")
        dates_and_counts = combine_multiple_data(
            dates_and_counts_a,
            dates_and_counts_b,
            combine_type=combine_type,
        )
        logger.info(f"After combining, obtained {len(dates_and_counts)} items")
    else:
        dates_and_counts = dates_and_counts_a

    pairings = []

    for i in range(num_randomizations):
        logger.info("Getting train/val split")
        if train_sequences_pre:
            logger.info("Using precomputed train/val/test splits")
            logger.info(
                f"{len(train_sequences_pre)} preliminary train sequences "
                f"{len(val_sequences_pre)} preliminary val sequences "
                f"{len(test_sequences_pre)} preliminary test sequences "
            )
            train_sequences, val_sequences, test_sequences = train_val_split_precalculated(
                dates_and_counts,
                train_sequences_pre,
                val_sequences_pre,
                test_sequences_pre,
            )
        else:
            logger.info("Determining new train/val/test splits")
            train_sequences, val_sequences, test_sequences = train_val_split(
                dates_and_counts,
                train_seqs_per_bucket,
                val_seqs_per_bucket,
                test_seqs_per_bucket,
                randshuffle_functor,
            )
        logger.info(f"Got {len(train_sequences)} training, {len(val_sequences)} val sequences, and {len(test_sequences)} test sequences")
        logger.info("Creating train pairings")
        train_pairings = list(
            get_pairings(
                train_sequences,
                max_sequence_comparisons=max_sequence_comparisons
            )
        )
        logger.info("Creating validation pairings")

        if val_sequences:
            val_pairings = list(
                get_pairings(
                    val_sequences,
                    max_sequence_comparisons=max_sequence_comparisons
                )
            )
        else:
            val_pairings = []

        logger.info(f"Got {len(train_pairings)} train and {len(val_pairings)} val pairings")

        if test_sequences:
            logger.info("Creating test pairings")
            test_pairings = list(
                get_pairings(
                    test_sequences,
                    max_sequence_comparisons=max_sequence_comparisons
                )
            )
            logger.info(f"Got {len(test_pairings)} test pairings")

        if control_locations:
            validate_helper = partial(
                validate_pairings,
                src_a=dates_and_counts_a,
                src_b=dates_and_counts_b,
                exact_test_cutoff=exact_test_cutoff,
                p_value=p_value,
                use_chi2=use_chi2,
                max_p_diff=max_p_diff,
            )

            if max_to_verify and len(train_pairings) > max_to_verify:
                train_pairings = random.sample(train_pairings, max_to_verify)

            if max_to_verify and len(val_pairings) > max_to_verify:
                val_pairings = random.sample(val_pairings, max_to_verify)

            if test_sequences:
                if max_to_verify and len(test_pairings) > max_to_verify:
                    test_pairings = random.sample(test_pairings, max_to_verify)

            train_pairings = validate_helper(train_pairings)

            if val_pairings:
                val_pairings = validate_helper(val_pairings)

            if test_sequences:
                test_pairings = validate_helper(test_pairings)

        if test_sequences:
            logger.info(f"After validation, we have train={len(train_pairings)}, "
                f"val={len(val_pairings)}, test={len(test_pairings)}")
        else:
            logger.info(f"After validation, we have train={len(train_pairings)}, "
                f"val={len(val_pairings)}")

        if (
            n_trailing_weeks_for_rare and
            min_count_high_freq and
            max_count_rare and
            max_pairings_for_one_seq
        ):
            logger.info("Adding freq-rare pairs")

            frequent_sequences, rare_sequences = get_hi_lo_pairings(
                df, n_trailing_weeks=n_trailing_weeks_for_rare,
                min_count=min_count_high_freq,
                max_total=max_count_rare,
            )

            train_pairings, val_pairings, test_pairings = add_hi_lo_pairings(
                train_pairings,
                val_pairings,
                test_pairings,
                frequent_sequences,
                rare_sequences,
                train_seqs_per_bucket,
                val_seqs_per_bucket,
                test_seqs_per_bucket,
                max_pairings_for_one_seq=max_pairings_for_one_seq,
            )

            logger.info(f"After adding freq-rare pairs, we have train={len(train_pairings)}, "
                f"val={len(val_pairings)}, test={len(test_pairings)}")

        if n_random_mutations_for_negatives:
            logger.info("Adding fake sequence examples")
            all_sequences, all_mutations = None, None

            for seq_pairing, neg_trials_per_seq in zip(
                [train_pairings, val_pairings, test_pairings],
                [
                    n_negative_trials_per_sequence_train,
                    n_negative_trials_per_sequence_val,
                    n_negative_trials_per_sequence_val,
                ]
            ):
                if seq_pairing:
                    all_sequences, all_mutations = add_fake_sequences_top(
                        df=df,
                        sequence_pair_list=seq_pairing,
                        n_random_mutations_for_negatives=n_random_mutations_for_negatives,
                        n_negative_trials_per_sequence=neg_trials_per_seq,
                        all_sequences=all_sequences,
                        all_mutations=all_mutations,
                    )

            logger.info(f"After adding fake sequences, we have train={len(train_pairings)}, "
                f"val={len(val_pairings)}, test={len(test_pairings)}")

        if test_sequences:
            pairings.append((train_pairings, val_pairings, test_pairings))
        else:
            pairings.append((train_pairings, val_pairings, []))

    if len(pairings) == 1:
        return pairings.pop()

    return pairings


if __name__ == "__main__":
    _GET_WEEKLY_COUNTS = True

    parser = argparse.ArgumentParser("Write sequence pairings to file")

    parser.add_argument(
        "--tsv",
        help="TSV source file",
        required=True,
    )

    parser.add_argument(
        "--availability_last_date",
        help="Last date of availability",
        required=True,
    )

    parser.add_argument(
        "--discovery_last_date",
        help="Last date for sequence discovery to use",
        required=False,
    )

    parser.add_argument(
        "--train_seqs_per_bucket",
        help="Number of training sequences per bucket",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--val_seqs_per_bucket",
        help="Number of validation sequences per bucket",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--test_seqs_per_bucket",
        help="Number of testing sequences per bucket",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--datefield",
        help="Date field to sort tsv file",
        default="Submission date",
    )

    parser.add_argument(
        "--protein",
        help="Protein for which we are collecting data",
        default="Spike",
    )

    parser.add_argument(
        "--period_length",
        help="Length of a period",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--output_prefix",
        help="Prefix of output file",
        required=True,
    )

    parser.add_argument(
        "--bucketization",
        help="How to bucketize data (_FRACTIONS data)",
        required=False,
    )

    parser.add_argument(
        "--primary_locations",
        help="Primary location strings to look for",
        default="Europe",
    )

    parser.add_argument(
        "--control_locations",
        help="Control locations to look for",
        required=False,
    )

    parser.add_argument(
        "--num_randomizations",
        help="Create multiple randomizations for cross-validation",
        type=int,
    )

    parser.add_argument(
        "--exact_test_cutoff",
        help="Cutoff to perform exact test",
        default=1e-2,
        type=float,
    )

    parser.add_argument(
        "--p_value",
        help="P-value below which null-hyptothesis is rejected. If not provided, statistical testing is not used",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--use_chi2",
        help="Use chi-squared test to speedup validation",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--max_p_diff",
        help="Max difference in the fraction between two regional occurrence probabilities",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--min_lead_time",
        help="Minimum lead time",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--n_trailing_weeks_for_rare",
        help="Number of zero occurrence trailing weeks for classifying as rare",
        type=int,
    )

    parser.add_argument(
        "--min_count_high_freq",
        help="Minimum sequence count tobe classified as high frequency",
        type=int,
    )

    parser.add_argument(
        "--max_count_rare",
        help="Maximum sequence count to be classified as rare sequence",
        type=int,
    )

    parser.add_argument(
        "--max_pairings_for_one_seq",
        help="In freq-rare pairing, how many rare sequences to pair with one frequent sequences",
        type=int,
    )

    parser.add_argument(
        "--n_random_mutations_for_negatives",
        help="To create fake sequences, set this value. Number of randomly selected mutations to add",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--n_negative_trials_per_sequence_train",
        help="Number of negative sequences to create for each train sequence",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--n_negative_trials_per_sequence_val",
        help="Number of negative sequences to create for each val/test sequence",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--train_sequences",
        help="Precomputed set of train sequences",
        default=None,
    )

    parser.add_argument(
        "--val_sequences",
        help="Precomputed set of val sequences",
        default=None,
    )

    parser.add_argument(
        "--test_sequences",
        help="Precomputed set of test sequences",
        default=None,
    )

    parser.add_argument(
        "--combine_type",
        help="How to combine sequences from primary and control locations",
        default="intersection",
    )

    parser.add_argument(
        "--max_to_verify",
        help="If there are too many sequence pairs, only verify a max number of them",
        default=1000000,
        type=int,
    )

    parser.add_argument(
        "--max_sequence_comparisons",
        help="Maximum number of sequences to compare one sequence to",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--max_train",
        help="Maximum number of training examples to output",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--max_val",
        help="Maximum number of validation examples to output",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--max_test",
        help="Maximum number of test examples to output",
        default=None,
        type=int,
    )
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    if args.bucketization:
        with open(args.bucketization, "r") as fhandle:
            bucketization = json.load(fhandle)
        _FRACTIONS = [CompetitionBrackets(**x) for x in bucketization]

    if args.min_lead_time:
        _MIN_LEAD_TIME = args.min_lead_time

    primary_locations = args.primary_locations.split(",")

    if not args.control_locations:
        control_locations = None
        exclude_control = False
    else:
        if args.control_locations.startswith("~"):
            control_locations = args.control_locations[1:].split(",")
            exclude_control = True
        else:
            control_locations = args.control_locations.split(",")
            exclude_control = False

    logger.info("Obtaining pairs")

    pairings_results = get_training_pairs_from_tsv(
        tsv=args.tsv,
        availability_last_date=args.availability_last_date,
        discovery_last_date=args.discovery_last_date,
        primary_locations=primary_locations,
        train_seqs_per_bucket=args.train_seqs_per_bucket,
        val_seqs_per_bucket=args.val_seqs_per_bucket,
        test_seqs_per_bucket=args.test_seqs_per_bucket,
        datefield=args.datefield,
        protein=args.protein,
        period_length=args.period_length,
        control_locations=control_locations,
        exclude_control=exclude_control,
        num_randomizations=args.num_randomizations,
        exact_test_cutoff=args.exact_test_cutoff,
        p_value=args.p_value,
        use_chi2=args.use_chi2,
        max_p_diff=args.max_p_diff,
        n_trailing_weeks_for_rare=args.n_trailing_weeks_for_rare,
        min_count_high_freq=args.min_count_high_freq,
        max_count_rare=args.max_count_rare,
        max_pairings_for_one_seq=args.max_pairings_for_one_seq,
        n_random_mutations_for_negatives=args.n_random_mutations_for_negatives,
        n_negative_trials_per_sequence_train=args.n_negative_trials_per_sequence_train,
        n_negative_trials_per_sequence_val=args.n_negative_trials_per_sequence_val,
        train_sequences_pre=args.train_sequences,
        val_sequences_pre=args.val_sequences,
        test_sequences_pre=args.test_sequences,
        combine_type=args.combine_type,
        max_to_verify=args.max_to_verify,
        max_sequence_comparisons=args.max_sequence_comparisons,
    )

    def write_helper(fname: str, pairings: list, max_items: Optional[int] = None) -> None:
        if max_items and len(pairings) > max_items:
            pairings = random.sample(pairings, max_items)

        with open(fname, "w") as fhandle:
            for item in pairings:
                fhandle.write(json.dumps(item._asdict()) + "\n")

    logger.info("Writing to output")
    
    if type(pairings_results) is list:
        for i, (train_pairings, val_pairings, test_pairings) in enumerate(pairings_results):
            write_helper(f"{args.output_prefix}_{i}.train.json", train_pairings, args.max_train)
            write_helper(f"{args.output_prefix}_{i}.val.json", val_pairings, args.max_val)
            if test_pairings:
                write_helper(f"{args.output_prefix}_{i}.test.json", test_pairings, args.max_test)
    else:
        train_pairings, val_pairings, test_pairings = pairings_results
        write_helper(f"{args.output_prefix}.train.json", train_pairings, args.max_train)
        write_helper(f"{args.output_prefix}.val.json", val_pairings, args.max_val)
        if test_pairings:
            write_helper(f"{args.output_prefix}.test.json", test_pairings, args.max_test)
