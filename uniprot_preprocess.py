# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from typing import Generator, List, Union, Tuple, Set
from collections import namedtuple
import re
import tqdm
import math
import random
import logging
import pickle
import argparse
import gzip
import csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(name=__file__)

"""
UniRef header format:
>UniqueIdentifier ClusterName n=Members Tax=TaxonName TaxID=TaxonIdentifier RepID=RepresentativeMember

Example:
>UniRef50_Q9K794 Putative AgrB-like protein n=2 Tax=Bacillus TaxID=1386 RepID=AGRB_BACHD

Source:
https://www.uniprot.org/help/fasta-headers
"""
HeaderItems = namedtuple("HeaderItems", [
    "uniref_id", "cluster_name", "cluster_size",
    "taxonomy_name", "taxonomy_id", "representative_member"])


def read_fasta_headers(fastafile: str) -> Generator[HeaderItems, None, None]:
    header_fmt = re.compile(">([A-Za-z0-9_-]+) (.*?) n=([0-9]+) Tax=(.*?) TaxID=(.*?) RepID=(.*)")

    with gzip.open(fastafile, "r") as fhandle:
        # Note: archived xml->fasta converted files contain byte literals
        # rather than string literals
        for line in map(lambda x: x.decode("utf-8"), fhandle):
            if line[0] == ">":
                header = line.strip()
                try:
                    header_items = HeaderItems(*header_fmt.match(header).groups())
                except AttributeError as e:
                    logger.error(f"Bad header: {line.strip()}")
                    raise e
                yield header_items


def pickle_helper(items: list, filename: str) -> None:
    with open(filename, "wb") as fhandle:
        pickle.dump(items, fhandle)


def cluster_processing(
    header_reader: Generator[HeaderItems, None, None],
    output_prefix: str,
    train_val_test_frac: (0.98, 0.01, 0.01),
) -> None:
    clusters = set()

    logger.info("Collecting clusters and ID->cluster mappings")

    with open(f"{output_prefix}_id_cluster_assignment.csv", "w", newline="") as fhandle:
        writer = csv.DictWriter(fhandle, fieldnames=["UniRefID", "ClusterName"])
        writer.writeheader()

        for item in tqdm.tqdm(header_reader, desc="Reading headers"):
            clusters.add(item.cluster_name)
            writer.writerow({"UniRefID": item.uniref_id, "ClusterName": item.cluster_name})

    logger.info(f"Collected {len(clusters)} clusters")
    logger.info("Obtaining train-val-test cluster splits")
    clusters = list(clusters)
    train_val_test_frac = [i / sum(train_val_test_frac) for i in train_val_test_frac]
    n_train = math.ceil(train_val_test_frac[0] * len(clusters))
    n_val = math.floor(train_val_test_frac[1] * len(clusters))
    random.shuffle(clusters)
    train_clusters = clusters[:n_train]
    val_clusters = clusters[n_train: n_train + n_val]
    test_clusters = clusters[n_train + n_val: ]
    logger.info(f"Obtained train={len(train_clusters)}, val={len(val_clusters)}, test={len(test_clusters)}")
    logger.info("Writing cluster splits to disk")
    pickle_helper(train_clusters, output_prefix + ".train.clusters")
    pickle_helper(val_clusters, output_prefix + ".val.clusters")
    pickle_helper(test_clusters, output_prefix + ".test.clusters")


def main(args: argparse.Namespace):
    header_reader = read_fasta_headers(args.fasta)
    cluster_processing(
        header_reader,
        args.output_prefix,
        [float(x) for x in args.train_val_test.split(",")],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process uniprot fasta headers to produce train/val/test splits")
    parser.add_argument(
        "--fasta", help="Uniprot fasta file", required=True)
    parser.add_argument(
        "--output_prefix", help="Prefix of output files", required=True)
    parser.add_argument(
        "--train_val_test", help="Train val test split", default="0.98,0.01,0.01")
    parser.add_argument(
        "--seed", help="Random seed", default=42, type=int)
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
