# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import utils
import pandas_utils
from pysam import FastaFile
import tqdm
from collections import defaultdict
from collections.abc import Iterable
import pandas
from functools import lru_cache
from argparse import ArgumentParser, Namespace
import re
import logging

logging.basicConfig(level=logging.INFO)


@lru_cache(maxsize=None)
def get_full_sequence(mutations: str, reference: str) -> str:
    return utils.get_full_sequence(mutations, reference)


def run_test(mut_seqs: pandas.DataFrame, fastafile: str, reference: str) -> dict:
    ins_found = False
    del_found = False
    num_sequences_validated = 0
    total_sequences = 0

    with FastaFile(fastafile) as fhandle:
        accession_map = {}
        for item in fhandle.references:
            res = re.findall("\|(EPI_ISL_\d+)\|", item)
            if res:
                accession_map[res[0]] = item
        
        for idx, item in tqdm.tqdm(mut_seqs.iterrows()):
            total_sequences += 1
            accession = item["Accession ID"]
            spike = item["SpikeMutations"]

            if "stop" in spike:
                continue

            if accession not in accession_map:
                continue

            full_seq_imputed = get_full_sequence(spike, reference)
            actual_seq = fhandle.fetch(accession_map[accession])
            if not utils.verify_sequences(actual_seq, full_seq_imputed):
                logging.warning(f"{accession}: Cannot verify {actual_seq} vs {full_seq_imputed}")
                continue
            ins_found = ins_found or "ins" in spike
            del_found = del_found or "del" in spike
            num_sequences_validated += 1

    assert(num_sequences_validated > 0)
    assert(ins_found and del_found), f"ins_found={ins_found}, del_found={del_found}"

    logging.info(f"Validated {num_sequences_validated} sequences out of a possible {total_sequences}")


def test_main(args: Namespace):
    data = pandas_utils.read_data_frame_cached(args.tsv)

    with open(args.ref) as fhandle:
        reference = fhandle.read().strip()

    run_test(data, args.fasta, reference)


def test_simple():
    reference = "ACGACTZBADAC"
    mutations = "G3del,A4del,T6G,ins7XX"
    expected = "AC--CGZXXBADAC".replace("-", "")
    res = get_full_sequence(mutations, reference)
    assert(res == expected), f"exp={expected}, res={res}"
    print("Test test_simple passed!")



if __name__ == "__main__":
    test_simple()
    parser = ArgumentParser(description="Test mutation imputation")

    parser.add_argument("--tsv", help="Variants file", required=True)
    parser.add_argument("--ref", help="Reference file", required=True)
    parser.add_argument("--fasta", help="Fasta file", required=True)

    args = parser.parse_args()

    test_main(args)
