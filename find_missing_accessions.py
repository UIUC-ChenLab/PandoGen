# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
from utils import fasta_serial_reader, FastaItem
import os
import pandas
import logging
import tqdm
from typing import Generator


def get_missing_items(fastaname: str, handles_to_keep: set) -> Generator[FastaItem, None, None]:
    with open(fastaname, "r") as fhandle:
        current_record = None
        current_header = None
        num_yields = 0
        bar = tqdm.tqdm(desc="Reading Fasta")

        for line in fhandle:
            if line.startswith(">"):
                bar.update(1)
                header = line[1:].strip()
                if header in fastaname:
                    if current_record:
                        yield FastaItem(header=current_header, sequence=current_record)
                        num_yields += 1

                    current_record = ""
                    current_header = header
                else:
                    if current_record:
                        yield FastaItem(header=current_header, sequence=current_record)
                        num_yields += 1

                    current_record = None
                    current_header = None
            elif current_header is not None:
                current_record += line.strip()

            if num_yields >= len(handles_to_keep):
                break

        if current_record:
            yield FastaItem(header=current_header, sequence=current_record)
            num_yields += 1

        bar.close()

        if num_yields != len(handles_to_keep):
            logging.error(f"Could not find all handles. Expected = {len(handles_to_keep)}, got = {num_yields}")


@dataclass
class Arguments:
    variants: str = field(
        metadata={"help": "Variants tsv file"}
    )

    metadata: str = field(
        metadata={"help": "Metadata TSV file"}
    )

    pangolin_rundir: str = field(
        metadata={"help": "Pangolin run directory"}
    )

    output_merged: str = field(
        metadata={"help": "Merged output CSV file"}
    )


def get_accession_to_name_map(df_variants: pandas.DataFrame, df_meta: pandas.DataFrame) -> dict:
    joined = df_meta.join(
        df_variants.set_index("Accession ID"),
        on="Accession ID",
        how="right",
        lsuffix=":meta",
        rsuffix=":variants",
    )
    joined = joined.dropna(subset=["Accession ID"])
    l1 = len(joined)

    joined = joined.dropna(subset=["Virus name"])
    l2 = len(joined)

    logging.info(
        f"Original joined length = {l1}, filtered length = {l2}"
    )

    return joined


def read_all_results(pangolin_rundir: str) -> pandas.DataFrame:
    dfs = []

    for d, _, fs in os.walk(pangolin_rundir):
        if any(os.path.splitext(f)[-1] == ".fa" and os.path.split(f)[1] != "unrunnable.fa" for f in fs):
            continue

        for f in fs:
            if os.path.splitext(f)[-1] == ".csv":
                fullpath = os.path.join(d, f)
                logging.info(f"Getting results file {fullpath}")
                df = pandas.read_csv(fullpath)
                dfs.append(df)

    all_results = pandas.concat(dfs, ignore_index=True)
    logging.info(f"Undeduplicated length = {len(all_results)}")
    all_results = all_results.drop_duplicates(subset=["taxon"], keep="last")
    logging.info(f"Deduplicated length = {len(all_results)}")
    return all_results


def get_assigned(df_meta_variants: pandas.DataFrame, df_results: pandas.DataFrame) -> pandas.DataFrame:
    df_results = df_results.assign(VirusName=df_results.taxon.apply(lambda x: x.split("|")[0]))
    df_joined = df_meta_variants.join(df_results.set_index("VirusName"), on="Virus name", how="left", rsuffix=":Pangolin_rerun")
    df_without_assignment = df_joined[df_joined["lineage"].isna()]

    logging.info(f"Original df_meta_variants length = {len(df_meta_variants)}, joined with results, length = {len(df_joined)}, without assignment length = {len(df_without_assignment)}")

    return df_joined, df_without_assignment


def get_pct_mismatch(df: pandas.DataFrame) -> None:
    def get_mismatch_helper(d: pandas.DataFrame) -> pandas.DataFrame:
        return d[(d["Pango lineage:variants"] != d["lineage"])]

    df_mismatching = get_mismatch_helper(df)
    logging.info(
        f"Out of {len(df)} items, {len(df_mismatching)} items have mismatching Pango lineages on rerun")

    df_nona = df.dropna(subset=["Pango lineage:variants", "lineage"])
    df_mismatching_nona = get_mismatch_helper(df_nona)
    logging.info(
        f"(NONA) Out of {len(df_nona)} items, {len(df_mismatching_nona)} items have mismatching Pango lineages"
    )
    


def main(args):
    logging.info("Reading variants tsv file")
    df_variants = pandas.read_csv(args.variants, low_memory=False, sep="\t")
    df_variants = df_variants.dropna(subset=["Accession ID", "Host"])
    df_variants = df_variants[df_variants["Host"].isin(["Human", "human"])]

    logging.info("Reading metadata file")
    df_meta = pandas.read_csv(args.metadata, sep="\t", low_memory=False)
    df_meta = df_meta.dropna(subset=["Accession ID", "Host", "Virus name"])
    df_meta = df_meta[df_meta["Host"].isin(["Human", "human"])]

    logging.info("Mapping Accession ID to Virus name")
    df_meta_variants = get_accession_to_name_map(df_variants, df_meta)

    logging.info("Reading all pangolin rerun results")
    df_results = read_all_results(args.pangolin_rundir)

    logging.info(
        "Creating map of old and new assignments, and shortlisting cases without results")
    df_with_old_new_assignments, df_without_reassignment = get_assigned(df_meta_variants, df_results)

    logging.info("Getting mismatch numbers")

    get_pct_mismatch(df_with_old_new_assignments)

    logging.info("Writing merged csv")

    df_with_old_new_assignments.to_csv(args.output_merged, sep="\t", index=False)

    wo_assignment_name = os.path.splitext(args.output_merged)[0] + ".without_assignment.csv"
    df_without_reassignment.to_csv(wo_assignment_name, sep="\t", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = HfArgumentParser((Arguments, ))
    args,  = parser.parse_args_into_dataclasses()
    main(args)
