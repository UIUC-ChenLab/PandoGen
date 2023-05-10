# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
"""
Prepare uniref100 and uniref90 to uniref50 mappings
"""
import xmlschema
import io
from typing import Generator, List, Optional
import pysam
from collections.abc import Iterable
from collections import defaultdict
import tqdm
from argparse import ArgumentParser, Namespace
import pickle
import os
import glob
import hashlib
import numpy as np
import math
import logging

logger = logging.getLogger(__file__)


def get_document_name(header: List[str]) -> str:
    assert(header[0].startswith("<?xml version")), "xml version missing in header"
    return header[1].split()[0][1:]


def xml_reader(
    filename: str,
    schema_text: str,
    local_rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Read UniRef XML file
    """
    schema = xmlschema.XMLSchema(schema_text)
    counter = 0

    with open(filename, "r") as fhandle:
        header = None
        buffer = []
        document_name = None

        for line in fhandle:
            line = line.strip()
            if line.startswith("<entry"):
                if not header:
                    header = list(buffer)
                    document_name = get_document_name(header)
                    buffer.clear()

            buffer.append(line)

            if line.endswith("</entry>"):
                if (local_rank is None or
                    world_size is None or
                    counter % world_size == local_rank):
                    tail = f"</{document_name}>"
                    text = "\n".join(header + buffer + [tail])
                    textstr = io.StringIO(text)
                    yield schema.to_dict(textstr)
                buffer.clear()
                counter += 1


def get_id_value(list_of_dicts: List[dict], id_name: str) -> str:
    for item in list_of_dicts:
        if item["@type"] == id_name:
            return item["@value"]

    raise ValueError(f"Cannot find {id_name} in {list_of_dicts}")


def process_member(member: dict) -> None:
        uniref100_id = get_id_value(member['dbReference']['property'], "UniRef100 ID")
        uniref90_id = get_id_value(member['dbReference']['property'], "UniRef90 ID")
        return uniref100_id, uniref90_id


def uniref50_reader(xml_iterable: Iterable) -> Generator[tuple, None, None]:
    for item in xml_iterable:
        uniref100_keys = []
        uniref90_keys = []
        uniref50_key = item["entry"][0]["@id"]

        # Representative member
        u100_repr, u90_repr = process_member(item["entry"][0]["representativeMember"])
        uniref100_keys.append(u100_repr)
        uniref90_keys.append(u90_repr)

        # Other members
        if "member" in item["entry"][0]:
            for member in item["entry"][0]["member"]:
                u100, u90 = process_member(member)
                uniref100_keys.append(u100)
                uniref90_keys.append(u90)

        yield uniref50_key, uniref100_keys, uniref90_keys


def create_cluster_mappings(
    uniref50_iterable: Iterable,
    local_rank: Optional[int] = None,
    max_items_to_process: Optional[int] = None,
) -> tuple:
    """
    Create UniRef50 -> UniRef100/90 mappings
    """
    uniref100_mappings = dict()
    uniref90_mappings = dict()

    silent = not(local_rank is None or local_rank == 0)
    
    for i, (uniref50_key, uniref100_keys, uniref90_keys) in enumerate(tqdm.tqdm(
        uniref50_iterable, desc="Reading UniRef50 XML", disable=silent)):
        for k in uniref100_keys:
            uniref100_mappings[k] = uniref50_key
        for k in uniref90_keys:
            uniref90_mappings[k] = uniref50_key
        if max_items_to_process is not None and i + 1 == max_items_to_process:
            break
        
    return uniref100_mappings, uniref90_mappings


def get_output_names(output_prefix: str, local_rank: Optional[int] = None) -> str:
    if local_rank is not None:
        output_prefix = f"{output_prefix}_{local_rank}"

    return f"{output_prefix}.uniref100.pkl", f"{output_prefix}.uniref90.pkl"


def main(args: Namespace) -> None:
    xml_iterable = xml_reader(
        args.xml, args.schema, local_rank=args.local_rank, world_size=args.world_size)
    uniref_iterable = uniref50_reader(xml_iterable)

    uniref100_mappings, uniref90_mappings = create_cluster_mappings(
        uniref_iterable, local_rank=args.local_rank, max_items_to_process=args.max_to_read)

    uniref100_output_name, uniref90_output_name = get_output_names(
        args.output_prefix, args.local_rank)

    with open(uniref100_output_name, "wb") as fhandle:
        pickle.dump(uniref100_mappings, fhandle)

    with open(uniref90_output_name, "wb") as fhandle:
        pickle.dump(uniref90_mappings, fhandle)


def get_bucket_n_items(weights: list, n_buckets: int) -> list:
    total = sum(weights)
    weights = [i / total for i in weights]
    n_items = [math.floor(n_buckets * i) for i in weights]
    if sum(n_items) != n_buckets:
        n_items[-1] = n_buckets - sum(n_items[:-1])
        if n_items[-1] < 0:
            raise ValueError("Bucket resolution is not sufficient, increase resolution")
    return n_items


def get_index_of_item(item: int, n_items: list) -> int:
    n_items = [0] + n_items
    indexes = np.cumsum(n_items)
    bucket_id = None
    for i in range(len(indexes) - 1):
        if indexes[i] <= item < indexes[i + 1]:
            bucket_id = i
            break
    return bucket_id


class TrainValSplitter:
    class ClusterNotFoundError(Exception):
        pass
    """
    A way to do data splits based on cluster identity
    """
    def __init__(self, dictionary: dict, n_buckets: int = 10000):
        logger.info("Initializing indirection sampler")
        self.dictionary = dictionary
        self.n_buckets = n_buckets
        self.scorecard = defaultdict(int)

    def _get_cluster_name(self, uniref_id: str) -> str:
        if uniref_id in self.dictionary:
            return self.dictionary[uniref_id]
        raise TrainValSplitter.ClusterNotFoundError

    def __call__(self, array: list, weights: list, k: int, uniref_id: str) -> str:
        cluster_name = self._get_cluster_name(uniref_id)
        h = int(hashlib.sha256(cluster_name.encode("utf-8")).hexdigest()[-8:], 16) % self.n_buckets
        n_items = get_bucket_n_items(weights, self.n_buckets)
        bucket_id = get_index_of_item(h, n_items)
        if bucket_id is None:
            raise TrainValSplitter.ClusterNotFoundError
        res = array[bucket_id]
        self.scorecard[res] += 1
        return [res]


if __name__ == "__main__":
    parser = ArgumentParser(description="Map UniRef50 IDs to UniRef100 and UniRef90")

    parser.add_argument("--xml", help="UniRef50 XML", required=True)
    parser.add_argument("--schema", help="UniRef schema file", required=True)
    parser.add_argument("--output_prefix", help="Prefix of output file", required=True)
    parser.add_argument(
        "--max_to_read", help="Maximum number of items to read", default=None, type=int)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    try:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        args.local_rank = None
        args.world_size = None

    main(args)
