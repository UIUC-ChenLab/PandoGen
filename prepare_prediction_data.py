# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pandas_utils
from sars_cov2_data_processing import get_full_sequence
import argparse
import datetime
import logging

logger = logging.getLogger()


def main(args):
    logger.info("Reading file")
    df = pandas_utils.read_data_frame_cached(args.tsv, protein=args.protein, datefield=args.date_field)

    logger.info("Extracting mutation sequences")
    dates = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in args.date_range.split(",")]
    df = df[(df["ParsedDate"] >= dates[0]) & (df["ParsedDate"] <= dates[1])]
    mutations = set(x for x in df[f"{args.protein}Mutations"].tolist() if "stop" not in x)

    with open(args.ref) as fhandle:
        reference = fhandle.read().strip()

    logger.info("Preparing output file")
    fhandles = [open(f"{args.output_prefix}.{j}.fa", "w") for j in range(args.num_splits)]

    for i, m in enumerate(mutations):
        m_ = m.replace(",", "_")
        m_ = "REF" if not m_ else m_
        fhandle = fhandles[i % len(fhandles)]
        fhandle.write(f">{m_}\n")
        fhandle.write(get_full_sequence(m, reference) + "\n")
        fhandle.write("\n")

    for f in fhandles:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create prediction data")

    parser.add_argument(
        "--tsv",
        help="Variant tsv file",
        required=True,
    )

    parser.add_argument(
        "--date_range",
        help="Comma-separated date-range from which to obtain sequences for prediction",
        required=True,
    )

    parser.add_argument(
        "--output_prefix",
        help="Prefix of output file",
        required=True,
    )

    parser.add_argument(
        "--protein",
        help="Protein name",
        default="Spike",
    )

    parser.add_argument(
        "--ref",
        help="Reference sequence file",
        required=True,
    )

    parser.add_argument(
        "--date_field",
        help="Date field to sort by",
        default="Submission date",
    )

    parser.add_argument(
        "--num_splits",
        help="Split results into multiple files to launch parallel simulations",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    main(args)
