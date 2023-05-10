# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pandas
import os
import re
import logging
from argparse import ArgumentParser
from typing import Optional
from utils import _GISAID_REFERENCE, get_full_sequence, verify_sequences
import pysam

logger = logging.getLogger(name=__file__)


_COLUMN_REMANING = {
    "Pango lineage": "PangoLineage"
}


def read_data_frame(tsv: str, max_to_sample: int = -1) -> pandas.DataFrame:
    logger.info("Reading and filtering variants file")

    variants_data = pandas.read_csv(tsv, sep="\t", low_memory=False)
    variants_data = variants_data[variants_data["Host"].isin(["Human", "human"])]

    if max_to_sample > 0:
        logger.info("Subsampling")
        variants_data = variants_data.loc[
            np.random.choice(variants_data.index, size=max_to_sample, replace=False)]
        
    logger.info(
        "Filtered variants list with %d lines" % len(
            variants_data
        )
    )

    return variants_data


def mutation_from_string(mutation: str, protein: str = "Spike") -> list:
    splits = re.findall("%s_([A-Za-z]+[0-9]+[A-Za-z]+)" % protein, mutation)
    sorted_splits = sorted(splits, key=lambda x: int(re.findall("[A-Za-z]+([0-9]+)[A-Za-z]", x)[0]))
    return ",".join(sorted_splits)


def rename_columns(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=_COLUMN_REMANING)


def check_sequence_validity(mutations: str, reference: str, sequenced: str) -> bool:
    full_sequence_imputed = get_full_sequence(mutations, reference)
    return verify_sequences(sequenced, full_sequence_imputed)


def reconstruction_label(
    df: pandas.DataFrame,
    fasta: str,
    protein: str = "Spike",
) -> pandas.DataFrame:
    """
    Attach a label to a sequence indicating whether it
    can be reconstructed from mutations
    """
    with pysam.FastaFile(fasta) as fhandle:
        accession_map = {}
        for item in fhandle.references:
            res = re.findall("\|(EPI_ISL_\d+)\|", item)
            if res:
                accession_map[res[0]] = item
        reference = fhandle.fetch(accession_map[_GISAID_REFERENCE])
        def helper(row):
            isolate = row["Accession ID"]
            if isolate not in accession_map:
                return False
            sequenced = fhandle.fetch(accession_map[isolate])
            return check_sequence_validity(row[f"{protein}Mutations"], reference, sequenced)
        df = df.assign(**{
            f"reconstruction_success_{protein}": df.apply(helper, 1)
        })
    return df


def preprocess_df(
    df: pandas.DataFrame,
    protein: str ="Spike",
    fasta: Optional[str] = None,
    impute_fake_label: bool = False,
) -> pandas.DataFrame:
    logger.info(f"Imputing {protein} mutations")

    df["AA Substitutions"].fillna("", inplace=True)  # Pandas replaces '' with NaN when storing, so we reverse that
    df = df.dropna(subset=["AA Substitutions", "Pango lineage"])

    # Impute Mutations
    df = df.assign(**
        {
            f"{protein}Mutations": df["AA Substitutions"].apply(
                lambda x: mutation_from_string(x, protein=protein)
            )
        }
    )

    if fasta or impute_fake_label:
        logger.info("Checking whether sequences can be reconstructed")
        if fasta:
            df = reconstruction_label(df, fasta, protein)
        if impute_fake_label:
            df = df.assign(**{
                f"reconstruction_success_{protein}": True
            })

    # # Add parsed date objects
    # logger.info("Adding dates")
    # df = df[df["Collection date"].str.match("[0-9]+-[0-9]+-[0-9]+")]
    # df = df.assign(ParsedDate=pandas.to_datetime(df["Collection date"]))

    # Get better column names
    df = rename_columns(df)

    return df


def get_cached_name(tsv: str, protein: str = "Spike") -> str:
    basename = os.path.splitext(tsv)[0]
    tsv_pre = basename + f".{protein}.preprocessed.tsv"
    return tsv_pre


def add_dates(df: pandas.DataFrame, datefield: str = "Submission date") -> pandas.DataFrame:
    # df = df[df[datefield].str.match("\d+-\d+-\d+")]
    df = df.assign(ParsedDate=pandas.to_datetime(df[datefield]))
    return df


def read_data_frame_cached(
    tsv: str,
    no_cache: bool = False,
    datefield: str = "Submission date",
    protein: str = "Spike",
    fasta: Optional[str] = None,
    impute_fake_label: bool = False,
) -> pandas.DataFrame:
    tsv_pre = get_cached_name(tsv, protein)

    if os.path.exists(tsv_pre) and not no_cache:
        df = pandas.read_csv(tsv_pre, sep="\t", low_memory=False)
    else:
        df = preprocess_df(
            read_data_frame(tsv),
            protein=protein,
            fasta=fasta,
            impute_fake_label=impute_fake_label
        )
        df.to_csv(tsv_pre, sep="\t")

    df = add_dates(df, datefield=datefield)
    df.SpikeMutations.fillna("", inplace=True)  # Pandas replaces '' with NaN when storing, so we reverse that
    logger.info(f"Final DataFrame has {len(df)} items")
    assert(not df.empty)

    return df


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a preprocessed variants file")
    parser.add_argument("--tsv", help="Input TSV file", required=True)
    parser.add_argument("--fasta", help="Fasta file of spike protein sequences", required=False)
    parser.add_argument("--protein", help="Protein name", default="Spike")
    parser.add_argument("--date_field", help="Which date field to use", default="Submission date")
    parser.add_argument("--impute_fake_label", help="Impute fake reconstruction_success label", default=False, action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    assert(not(args.fasta and args.impute_fake_label)), "Cannot have both these inputs"

    tsv_pre = get_cached_name(args.tsv, args.protein)

    if os.path.exists(tsv_pre):
        logger.info("%s exists, deleting" % tsv_pre)
        os.remove(tsv_pre)

    logger.info("Reading and caching file")

    read_data_frame_cached(
        args.tsv,
        no_cache=True,
        datefield=args.date_field,
        protein=args.protein,
        fasta=args.fasta,
        impute_fake_label=args.impute_fake_label,
    )
