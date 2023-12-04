# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
"""
Create validation data for competition models that uses classification prediction
for a given number of weeks
"""
import sys
import pandas_utils
import datetime
import pandas
import create_occurrence_buckets
import json
import utils
import logging
from argparse import ArgumentParser

logger = logging.getLogger(__file__)


def main(args):
    # Prepare DataFrame within training period
    logger.info("Reading data, filtering and adding columns")
    df = pandas_utils.read_data_frame_cached(args.tsv, datefield=args.datefield, protein=args.protein)
    df = df[df[f"reconstruction_success_{args.protein}"]]
    end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    df = df[df.ParsedDate <= end_date]
    min_date = df.ParsedDate.min()
    df_weeks = df.assign(week=df.apply(lambda x: (x.ParsedDate - min_date).days // 7, 1))
    max_week = df_weeks.week.max()

    # Get count data
    logger.info("Obtaining count data per sequence")
    df_weeks_counts = create_occurrence_buckets.get_counts(df_weeks)
    count_data = dict(zip(df_weeks_counts.index, df_weeks_counts))

    # Get discovery dates/weeks
    logger.info("Getting discovery weeks/dates of sequences")
    df_discovery = df_weeks.loc[df_weeks.groupby("SpikeMutations").ParsedDate.idxmin()]

    # Get spikes discovered in the first week immediately after
    # training sequences are obtained, within the training period
    # Note: max_week - lead_time + 1 is retained in the training data
    # See lines 476 in create_occurrence_buckets
    logger.info(
        "Getting spike sequences discovered in the week "
        "immediately succeeding the week with the last training sequences "
        "from within the training period"
    )
    df_discovery_last_week = df_discovery[
        (df_discovery.week == max_week - args.lead_time + 2)]
    spikes = df_discovery_last_week.SpikeMutations.tolist()

    # Get results dictionary
    logger.info("Obtaining results")
    spikes_with_counts = {
        key: sum(v[1] for v in count_data[key]) for key in spikes}

    # Post-process and write
    logger.info("Writing")
    with open(args.ref) as fhandle:
        ref = fhandle.read().strip()

    with open(args.output_prefix + ".json", "w") as fhandle:
        for item in spikes_with_counts.items():
            seq = utils.get_full_sequence(item[0], ref)
            fhandle.write(json.dumps({"seq": seq, "count": item[1]}) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = ArgumentParser(description="Create validation data for competition trainer")
    parser.add_argument("--tsv", help="Variants TSV file", required=True)
    parser.add_argument("--datefield", help="Date sort field", default="Submission date")
    parser.add_argument(
        "--protein", help="Protein for which we run experiments", default="Spike")
    parser.add_argument(
        "--end_date", help="End date of training period (inclusive)", required=True)
    parser.add_argument("--ref", help="Reference sequence path", required=True)
    parser.add_argument("--output_prefix", help="Prefix of output file", required=True)
    parser.add_argument("--lead_time", help="Lead time used for competition sequences",
        default=create_occurrence_buckets._MIN_LEAD_TIME, type=int)
    args = parser.parse_args()
    main(args)
