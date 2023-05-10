# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import subprocess
import argparse
import pandas_utils
import datetime
from utils import get_full_sequence
from typing import List
import shutil
import os
import logging
import tqdm
import json
import pysam

logger = logging.getLogger(__file__)


def parse_date(date_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")


def get_training_sequences(
    tsv: str,
    last_date: str,
    ref_file: str,
    datefield: str = "Submission date",
    protein: str = "Spike"
) -> list:
    logger.info("Reading TSV")

    df = pandas_utils.read_data_frame_cached(
        tsv,
        datefield=datefield,
        protein=protein,
    )

    logger.info("Filtering")
    df = df[df[f"reconstruction_success_{protein}"]]
    df_training_period = df[df.ParsedDate <= parse_date(last_date)]

    with open(ref_file, "r") as fhandle:
        ref = fhandle.read().strip()

    logger.info("Extracting training sequences")

    seqs = [
        get_full_sequence(m, ref) for m in tqdm.tqdm(set(df_training_period[f"{protein}Mutations"].tolist()), desc="Extracting sequences") \
            if "stop" not in m
    ]

    return seqs


def get_kmers(sequences: List[str], length: int):
    logger.info("Getting kmers")
    kmers = set()
    
    for seq in sequences:
        for i in range(len(seq) - length + 1):
            kmers.add(seq[i: i + length])
            
    return kmers


def get_per_sequence_novelty(sequence: str, ref_kmers: set, known_valid: bool = False) -> int:
    if not known_valid:
        if "[" in sequence or "]" in sequence:
            return None

        if "<" in sequence or ">" in sequence:
            return None

        if sequence.count("*") != 1:
            return None
    
    len_kmer = len(next(iter(ref_kmers)))
    
    n_novel = 0
    
    for kmer_pos in range(len(sequence) - len_kmer + 1):
        kmer = sequence[kmer_pos: kmer_pos + len_kmer]
        if kmer not in ref_kmers:
            n_novel += 1
        
    return n_novel


def get_per_sequence_novelty_for_sample(filename: str, ref_kmers: set) -> float:
    sequences = set()
    
    with open(filename, "r") as fhandle:
        for line in fhandle:
            items = json.loads(line)
            sequences.add(items["seq"])
            
    total_novel_kmers = 0
    total_valid_sequences = 0
    
    for seq in sequences:
        res = get_per_sequence_novelty(seq, ref_kmers)
        if res is not None:
            total_novel_kmers += res
            total_valid_sequences += 1
            
    return total_novel_kmers / total_valid_sequences, total_valid_sequences


def run_generation_and_get_distance(
    cmd: str,
    workdir: str,
    ref_kmers: set,
):
    if os.path.exists(workdir):
        raise ValueError(f"{workdir} exists!")

    os.makedirs(workdir)

    with open(os.path.join(workdir, "cmd.sh"), "w") as fhandle:
        cmdfile = fhandle.name
        output_prefix = os.path.join(workdir, "results")
        fhandle.write(f"{cmd} --output_prefix {output_prefix}")

    subprocess.run(["bash", cmdfile], check=True)

    novel_kmers, valid_sequences = get_per_sequence_novelty_for_sample(
        f"{output_prefix}.json", ref_kmers)

    shutil.rmtree(workdir)

    return novel_kmers, valid_sequences


def binary_search(
    parameter_name: str,
    parameter_values: tuple,
    expected_distance: float,
    expected_distance_tolerance: float,
    base_cmd: str,
    workdir: str,
    ref_kmers: set,
    max_iterations: int = 32,
):
    obtained_distance = float("inf")

    current_parameter_value = sum(parameter_values) / 2
    left, right = parameter_values
    n_iter = 0

    while n_iter < max_iterations:
        cmd = f"{base_cmd} --{parameter_name} {current_parameter_value}"

        logger.info(f"Running command prefix {cmd}")
        obtained_distance = run_generation_and_get_distance(cmd, workdir, ref_kmers)[0]

        logger.info(
            f"Obtained distance = {obtained_distance}, expected = {expected_distance}, parameter {parameter_name} = {current_parameter_value}"
        )

        lo_limit = expected_distance - expected_distance_tolerance
        hi_limit = expected_distance + expected_distance_tolerance

        logger.info(f"Checking {lo_limit} <= {obtained_distance} <= {hi_limit}")

        if lo_limit <= obtained_distance <= hi_limit:
            logger.info("Success!")
            break

        if obtained_distance > expected_distance:
            right = current_parameter_value
        else:
            left = current_parameter_value

        current_parameter_value = (left + right) / 2

        n_iter += 1


def main(args):
    if args.training_sequences:
        logger.info("Reading pre-provided training_sequences")
        with pysam.FastaFile(args.training_sequences) as fhandle:
            training_sequences = [fhandle.fetch(x) for x in fhandle.references]
    else:
        training_sequences = get_training_sequences(args.tsv, args.last_date, args.ref_file, args.datefield, args.protein)

    ref_kmers = get_kmers(training_sequences, length=args.k)

    with open(args.base_cmd) as fhandle:
        base_cmd = fhandle.read().strip()

    binary_search(
        args.parameter_name,
        tuple(float(x) for x in args.parameter_values.split(",")),
        expected_distance=args.expected_distance,
        expected_distance_tolerance=args.tolerance,
        base_cmd=base_cmd,
        workdir=args.workdir,
        ref_kmers=ref_kmers,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = argparse.ArgumentParser(
        description="Search for configuration that meets distance requirements"
    )

    parser.add_argument(
        "--tsv",
        help="TSV file (either provide this or --training_sequences)",
        required=False,
    )

    parser.add_argument(
        "--last_date",
        help="Last date in training period (either provide this or --training_sequences)",
        required=False,
    )

    parser.add_argument(
        "--ref_file",
        help="Reference file",
        required=False,
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
        "--k",
        help="K-mer length",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--base_cmd",
        help="File with base command",
        required=True,
    )

    parser.add_argument(
        "--parameter_name",
        help="Name of the parameter",
        required=True,
    )

    parser.add_argument(
        "--parameter_values",
        help="Parameter value limits (comma-separated)",
        required=True,
    )

    parser.add_argument(
        "--expected_distance",
        help="Expected distance",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--tolerance",
        required=True,
        help="Tolerance for the above",
        type=float,
    )

    parser.add_argument(
        "--workdir",
        help="Name of a non-existent path to write intermiediate data",
        required=True,
    )

    parser.add_argument(
        "--max_iterations",
        help="Maximum number of iterations",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--training_sequences",
        help="List of sequences in the training period",
        required=False,
    )

    args = parser.parse_args()
    main(args)
