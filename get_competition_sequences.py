# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pandas_utils
import pandas
from argparse import ArgumentParser
import datetime
from create_occurrence_buckets import prepare_tsv, get_counts, parse_date, accumulate_weekly_counts
import logging
import json

logger = logging.getLogger(__file__)


def get_sequences_in_timeframe(df: pandas.DataFrame, start: str, end: str, protein: str = "Spike") -> list:
    df = df.loc[df.groupby(f"{protein}Mutations")["ParsedDate"].idxmin()]
    df = df[(df.ParsedDate >= parse_date(start)) & (df.ParsedDate <= parse_date(end))]
    return df.SpikeMutations.tolist()


if __name__ == "__main__":
    parser = ArgumentParser(description="Get a list of sequences to make predictions for competition")

    parser.add_argument("--tsv", help="Pandas file", required=True)
    parser.add_argument("--start", help="Start date (inclusive)", required=True)
    parser.add_argument("--end", help="End date (inclusive)", required=True)
    parser.add_argument("--last_date", help="Last date assumption for the full data", required=True)
    parser.add_argument("--datefield", help="Date field to sort by", default="Submission date")
    parser.add_argument("--protein", help="Protein for which to make predictions", default="Spike")
    parser.add_argument("--output", help="Output file name", required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    df, _ = prepare_tsv(args.tsv, availability_last_date=args.last_date)
    df_counts = get_counts(df)
    counts_dict = {}

    for key, value in zip(df_counts.index, df_counts):
        counts_dict[key] = accumulate_weekly_counts(value)

    with open(args.output, "w") as fhandle:
        for item in get_sequences_in_timeframe(df, args.start, args.end):
            fhandle.write(json.dumps({"seq": item, "weekly_counts": counts_dict[item]}) + "\n")
