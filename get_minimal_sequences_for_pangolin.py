# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pandas
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
import re
import logging
import tqdm
from utils import fasta_serial_reader
import json

logger = logging.getLogger(__file__)


@dataclass
class Arguments:
    metadata: str = field(
        metadata={"help": "metadata.tsv file from GISAID"}
    )

    fasta: str = field(
        metadata={"help": "Fasta file"}
    )

    output: str = field(
        metadata={"help": "Output Fasta filename"}
    )


def canonicalize_mutations(mutations: str) -> str:
    if mutations[0] == "(":
        mutations = mutations[1:]

    if mutations[-1] == ")":
        mutations = mutations[:-1]

    if not mutations.strip():
        return ""

    mutations_orig = mutations

    mutations = [x.strip() for x in mutations.split(",")]

    def sort_key(mut_string: str) -> tuple:
        try:
            relevant_tuple = re.findall(r"([A-Za-z0-9]+)_[A-Za-z]+(\d+)[A-Za-z+]", mut_string).pop()
            return relevant_tuple[0], int(relevant_tuple[1])
        except IndexError as e:
            logging.warning(f"Cannot parse {mut_string}")
            return "", 0

    mutations = ",".join(sorted(mutations, key=sort_key))

    return mutations


def main(args: Arguments):
    logger.info("Reading source file")
    variants = pandas.read_csv(args.metadata, sep="\t", low_memory=False)
    variants = variants[variants["Host"].isin(["Human", "human"])]
    variants["AA Substitutions"].fillna("", inplace=True)
    variants.dropna(subset=["AA Substitutions", "Pango lineage", "Virus name", "Accession ID"])
    logger.info(f"Read file of length {len(variants)}")

    # logger.info("Removing low quality rows")
    # variants = variants[variants["Is high coverage?"]]
    # logger.info(f"Read file of length {len(variants)}")

    logger.info("Imputing additional columns")
    variants = variants.assign(
        canonical_mutation_representation=variants["AA Substitutions"].apply(canonicalize_mutations)
    )
    variants_full = variants
    variants = variants.assign(ParsedDate=pandas.to_datetime(variants["Submission date"]))
    variants = variants.loc[variants.groupby("canonical_mutation_representation")["ParsedDate"].idxmin()]
    valid_names = dict(zip(variants["Virus name"], variants["canonical_mutation_representation"]))

    logger.info(f"Obtained a total of {len(valid_names)} sequences for pango lineage assignment")

    with open(args.output, "w") as fhandle:
        for item in tqdm.tqdm(fasta_serial_reader(args.fasta), desc="Writing selected cases"):
            name = item.header.split("|")[0]
            if name in valid_names:
                fhandle.write(f">{item.header}\n{item.sequence}\n\n")

    logger.info("Writing name to canonical mapping")

    with open(os.path.splitext(args.output) + ".name_mapping.json", "w") as fhandle:
        json.dump(valid_names, fhandle, indent=4)

    logger.info("Writing enhanced metadata file")
    variants_full.to_csv(os.path.splitext(args.output) + ".metadata_enhanced.csv", sep="\t", index=False)

    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = HfArgumentParser((Arguments, ))
    args, = parser.parse_args_into_dataclasses()
    main(args)