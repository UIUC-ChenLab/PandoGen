# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import subprocess
import argparse
from utils import fasta_serial_reader, FastaItem
import math
from typing import Generator, List
import os
import logging
import tqdm
import shutil

logger = logging.getLogger(__file__)


def find_fasta_length(fasta_file: str):
    if os.path.exists(fasta_file + ".nlines"):
        with open(fasta_file + ".nlines", "r") as fhandle:
            n_lines = int(next(fhandle).strip())
        return n_lines

    with open(fasta_file, "r") as fhandle:
        counter = sum(1 for i in fhandle if i.startswith(">"))

    return counter


def fasta_reader_batched(fasta_file: str, batch_size: int) -> Generator[List[FastaItem], None, None]:
    batch = []

    for i in fasta_serial_reader(fasta_file, sep="|"):
        batch.append(i)
        if len(batch) >= batch_size:
            yield batch
            batch.clear()

    if batch:
        yield batch
        batch.clear()


def remove_handle_from_fasta(fasta_file: str, handle_to_remove: str):
    items = list(fasta_serial_reader(fasta_file, sep="|"))
    fasta_removed = os.path.join(os.path.split(fasta_file)[0], "unrunnable.fa")
    os.remove(fasta_file)
    found_last_unoffending = False
    offending_header = None

    with open(fasta_file, "w") as fhandle, open(fasta_removed, "a") as whandle:
        for item in items:
            if found_last_unoffending:
                offending_header = item.header
                whandle.write(f">{item.header}\n{item.sequence}\n")
                found_last_unoffending = False
                continue

            if item.header == handle_to_remove:
                found_last_unoffending = True

            fhandle.write(f">{item.header}\n{item.sequence}\n")

    return offending_header


def main(args):
    logger.info("Calculating fasta length")
    fasta_length = find_fasta_length(args.fasta)
    logger.info(f"Fasta has {fasta_length} entries")

    n_batches = math.ceil(fasta_length / args.batch_size)

    for i, batch in enumerate(tqdm.tqdm(fasta_reader_batched(args.fasta, args.batch_size), total=n_batches, desc="Processing batches")):
        if i - args.proc_num < 0 or (i - args.proc_num) % args.n_procs != 0:
            continue

        workdir = os.path.join(args.workdir, f"subworkdir{i}")
        os.makedirs(workdir)

        with open(os.path.join(workdir, "fasta_for_piece.fa"), "w") as fhandle:
            pangolin_input = fhandle.name
            outdir = workdir

            for item in batch:
                fhandle.write(f">{item.header}\n{item.sequence}\n")

        tempdir = os.path.join(outdir, "temp")

        command = [
            "pangolin",
            "--outdir", outdir,
            "--threads", str(args.n_pangolin_threads),
            "--temp", tempdir,
            pangolin_input,
        ]

        not_done = True

        while not_done:
            # Do pre-cleanup
            if os.path.exists(os.path.join(outdir, "logs")):
                os.remove(os.path.join(outdir, "logs"))

            if os.path.exists(os.path.join(outdir, "lineage_report.csv")):
                os.remove(os.path.join(outdir, "lineage_report.csv"))

            if os.path.exists(tempdir):
                shutil.rmtree(tempdir)

            # Run on the sub-batch
            with open(os.path.join(outdir, "logs"), "w") as fhandle:
                try:
                    subprocess.run(command, stdout=fhandle, stderr=fhandle, check=True)
                    os.remove(pangolin_input)
                    os.rmdir(tempdir)
                    not_done = False
                except subprocess.CalledProcessError:
                    logger.warning(f"Batch number {i} failed.")
                    partial_results = os.path.join(outdir, "lineage_report.csv")

                    if os.path.exists(partial_results):
                        with open(partial_results, "r") as fhandle:
                            for line in fhandle:
                                line = line.strip()
                            taxon_to_remove = line.split(",")[0]

                        logger.warning("Found offending fasta header, removing ...")
                        offending_header = remove_handle_from_fasta(pangolin_input, taxon_to_remove)
                        if offending_header:
                            logger.info("Removed offending header from pangolin input. Rerunning ...")
                        else:
                            logger.info(f"Cannot find offending header, please manually rerun batch {i}")
                            not_done = False
                    else:
                        logger.warning(f"Cannot find offending case, please manually rerun batch {i}")


    logger.info("Completed run")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = argparse.ArgumentParser(description="Piecewise Pangolin runner")
    parser.add_argument("--fasta", help="Fasta file input", required=True)
    parser.add_argument("--batch_size", help="Size of one batch", required=True, type=int)
    parser.add_argument("--n_procs", help="Number of processes that will be launched to analyze", type=int, default=1)
    parser.add_argument("--proc_num", help="What is the number of this proc", type=int, default=0)
    parser.add_argument("--workdir", help="Head working directory", required=True)
    parser.add_argument("--n_pangolin_threads", help="Number of threads for pangolin", type=int, default=16)
    args = parser.parse_args()
    main(args)
