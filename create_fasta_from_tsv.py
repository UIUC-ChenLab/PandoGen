# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from binary_search_parameter import get_training_sequences
from argparse import ArgumentParser
import logging


def main(args):
    sequences = get_training_sequences(
        args.tsv,
        args.last_date,
        args.ref_file,
        args.datefield,
        args.protein
    )

    with open(f"{args.output_prefix}.fa", "w") as fhandle:
        for i, seq in enumerate(sequences):
            fhandle.write(f">{i}\n{seq}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Create fasta file from TSV file")
    parser.add_argument(
        "--tsv",
        help="Variants TSV file",
        required=True,
    )
    parser.add_argument(
        "--last_date",
        help="Last date in training period (either provide this or --training_sequences)",
        required=True,
    )
    parser.add_argument(
        "--ref_file",
        help="Reference file",
        required=True,
    )
    parser.add_argument(
        "--datefield",
        default="Submission date",
        help="Sort field for date",
    )
    parser.add_argument(
        "--protein",
        default="Spike",
        help="Protein for which we are running search",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Prefix of output file",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    main(args)
