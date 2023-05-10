# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from dataclasses import dataclass
from functools import reduce
from operator import concat
from typing import Optional, Generator, List
import torch
import re
from collections import namedtuple


_AMBIGUOUS_CHARACTERS = "BXJZ"
_NONAMBIGUOUS_CHARACTERS = "ACDEFGHIKLMNPQRSTVWY"
_SPECIAL_CHARACTERS = "UO"  # https://pubmed.ncbi.nlm.nih.gov/15788401/; TBD: find a source for the letter
AMINO_ACIDS = list(_NONAMBIGUOUS_CHARACTERS + _AMBIGUOUS_CHARACTERS + _SPECIAL_CHARACTERS)
_MUTATION_PATTERN = re.compile(r"([A-Za-z]+)(\d+)([A-Za-z]+)")
_GISAID_REFERENCE = "EPI_ISL_402124"


FastaItem = namedtuple("FastaItem", ["header", "sequence"])


ambiguity_mapping = {
    "B": ["D", "N"],
    "J": ["L", "I"],
    "Z": ["Q", "E"],
}


def verify_sequences(sequenced: str, imputed: str):
    if len(sequenced) != len(imputed):
        return False

    return all(
        a == b or a == "X" or b in ambiguity_mapping.get(a, []) for a, b in \
            zip(sequenced, imputed)
    )


def mutation_positions_in_seq(mutations: str) -> List[int]:
    """
    Find mutation positions in a sequence given the mutations
    in the sequence. The mutations contain reference positions
    """
    mutations = mutations.strip()
    parsed_mutations = re.findall(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", mutations)

    if len(parsed_mutations) == 0:
        return []

    parsed_mutations = [(a, int(b) - 1, c) for a, b, c in parsed_mutations]
    clustered_mutations = []

    for p in parsed_mutations:
        if p[-1] == "del":
            if clustered_mutations and type(clustered_mutations[-1]) is list and \
                clustered_mutations[-1][-1][1] == p[1] - 1:
                clustered_mutations[-1].append(p)
            else:
                clustered_mutations.append([p])
        else:
            clustered_mutations.append(p)

    delta = 0
    seq_mut_positions = []

    for item in clustered_mutations:
        if type(item) is list:
            pos = item[0][1]
            seq_mut_positions.append((pos + delta - 1, pos + delta + 1))
            delta -= len(item)
        elif item[0] == "ins":
            pos = item[1]
            seq_mut_positions.append((pos + delta, pos + delta + len(item[-1]) + 2))
            delta += len(item[-1])
        else:
            pos = item[1]
            seq_mut_positions.append((pos + delta, pos + delta + 1))

    return seq_mut_positions


@dataclass
class SpecialTokens:
    start_of_sequence: str = "[CLS]"
    masked_segment_indicator: str = "[MASK]"
    end_of_sequence: str = "[SEP]"
    num_special_tokens: Optional[int] = None

    def __post_init__(self):
        self.num_special_tokens = len(set([
            self.start_of_sequence,
            self.masked_segment_indicator,
            self.end_of_sequence,
        ]))


_DEFAULT_SPECIAL_TOKENS = SpecialTokens()


def concat_lists(*args) -> list:
    return reduce(lambda a, b: concat(a, b), args, [])


def is_cuda(model: torch.nn.Module) -> bool:
    return next(model.parameters()).is_cuda


def get_full_sequence(mutations: str, reference: str) -> str:
    try:
        sequence = [list(x) for x in reference]
        mutations = mutations.strip()

        if len(mutations) > 0:
            for m in mutations.split(","):
                res = _MUTATION_PATTERN.match(m)
                if not res:
                    raise ValueError(f"Mutation {m} from {mutations} doesn't match mutation format")

                a, pos, b = res.groups()
                pos = int(pos)
                b = "" if b == "del" else b

                if a == "ins":
                    sequence[pos - 1] += b
                else:
                    sequence[pos - 1][0] = b
    except Exception as e:
        print(f"Cannot convert {mutations} for reference of length {len(reference)}")
        raise e

    return "".join(["".join(x) for x in sequence])


def fasta_serial_reader(fasta_file: str) -> Generator[FastaItem, None, None]:
    """
    Read a Fasta File serially
    """
    with open(fasta_file, "r") as fhandle:
        header = None
        collected = []

        for line in fhandle:
            line = line.strip()
            if line.startswith(">"):                
                if collected:
                    yield FastaItem(header, "".join(collected))
                    collected.clear()
                    header = None
                header = line[1:].split()[0]
            elif header and len(line) > 0:
                collected.append(line)

        if collected:
            yield FastaItem(header, "".join(collected))
 