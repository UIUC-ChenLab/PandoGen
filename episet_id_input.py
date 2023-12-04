# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pandas_utils
import argparse


def main(args):
    df = pandas_utils.read_data_frame_cached(
        args.tsv,
        datefield="Submission date",
        protein="Spike",
    )

    acc = df["Accession ID"].tolist()

    with open(args.output_prefix + ".txt", "w") as fhandle:
        for item in acc:
            fhandle.write(item + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EPISET ID creation inputs"
    )

    parser.add_argument(
        "--tsv",
        help="Variant surveillance file",
        required=True,
    )

    parser.add_argument(
        "--output_prefix",
        help="Prefix of output file",
        required=True,
    )

    args = parser.parse_args()
    main(args)
