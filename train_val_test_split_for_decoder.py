# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
"""
For training a pure decoder, we split uniref50 into train/val/test set
"""
import pysam
import random
import argparse
import math
import numpy as np
import tqdm
import logging

logger = logging.getLogger(__file__)


def main(args):
    types = ["train", "val", "test"]
    counters = {key: 0 for key in types}
    weights = np.array([args.train_weight, args.val_weight, args.test_weight])
    frac = weights / np.add.reduce(weights)

    logger.info(f"Splitting data at approximately the following fraction: {frac}")

    with pysam.FastaFile(args.fa) as fhandle, \
        open(args.output_prefix + ".train.fa", "w") as trainfa, \
            open(args.output_prefix + ".val.fa", "w") as valfa, \
                open(args.output_prefix + ".test.fa", "w") as testfa:

        handles = {
            "train": trainfa,
            "val": valfa,
            "test": testfa
        }

        for r in tqdm.tqdm(fhandle.references, desc="Processing FASTA"):
            t = np.random.choice(3, size=1, p=frac)[0]
            dtype = types[t]
            handles[dtype].write(f">{r}\n{fhandle.fetch(r)}\n\n")
            counters[dtype] += 1

    logger.info(f"Finished writing, distribution = {counters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a train/val/test split of a fasta")

    parser.add_argument("--fa", help="File name", required=True)
    parser.add_argument("--train_weight", help="Weight of train", required=True, type=float)
    parser.add_argument("--val_weight", help="Weight of val", required=True, type=float)
    parser.add_argument("--test_weight", help="Weight of test", required=True, type=float)
    parser.add_argument("--output_prefix", help="Prefix of output file", required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    main(args)

