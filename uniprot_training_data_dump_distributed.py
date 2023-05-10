# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import data_processing
import pysam
import pickle
import logging
import csv
import tqdm
from collections import defaultdict
import json
import argparse
import os
import math
import multiprocessing
from functools import partial
from typing import Callable, Generator, Union, Optional
from collections.abc import Iterable
import random
import copy
import datetime
import shutil
import torch
from uniref_cluster_preprocessing import TrainValSplitter, get_output_names
from utils import fasta_serial_reader
from itertools import islice
import glob

logger = logging.getLogger()


_TEST_RANDSAMPLE_FUNCTOR = None
_TEST_RANDINT_FUNCTOR = None
_TEST_RANDCHOICES_FUNCTOR = None


class RandChoicesFunctor:
    def __init__(self):
        pass

    def __call__(self, array, weights, k, uniref_id):
        if uniref_id in ["ABCD0", "ABCD2a"]:
            return ["train"]
        elif uniref_id in ["ABCD1"]:
            return ["test"]
        elif uniref_id in ["ABCD3"]:
            return ["val"]
        else:
            raise ValueError("Bad uniref_id")


class RandSampleFunctor:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return [2]


class RandIntFunctor:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return 3


def read_pickled_file(f: str) -> object:
    with open(f, "rb") as fhandle:
        data = pickle.load(fhandle)
    return set(data)


def split_sequences(
    full_sequence: str,
    max_length: int,
    remove_asterisk: bool = True,
) -> Generator[str, None, None]:
    if remove_asterisk:
        if full_sequence[-1] == "*":
            full_sequence = full_sequence[: -1]

    num_splits = math.ceil(len(full_sequence) / max_length)

    for i in range(num_splits):
        yield full_sequence[i * max_length: (i + 1) * max_length] 


def write_data(
    csvfile_name: str,
    addresses: defaultdict,
    disk_storage: dict,
    procnum: int,
    total: int,
) -> None:
    disk_storage = {key: data_processing.DiskStorage.load(value, mode="r+") for \
        key, value in disk_storage.items()}

    with open(csvfile_name, "r", newline="") as fhandle:
        reader = csv.DictReader(fhandle)

        for row in tqdm.tqdm(
            reader,
            disable=(procnum != 0),
            total=total,
            desc=f"Writing to disc (process {procnum})",
        ):
            encoder_seq = json.loads(row["encoder_seq"])
            decoder_seq = json.loads(row["decoder_seq"])
            metadata = {"UnIRefID": row["UniRefID"], "fragment": row["fragment"]}
            disk_storage[row["data_type"]] = (encoder_seq, decoder_seq, metadata)


def default_rand_choices_wrapper(
    array: list, weights: Union[list, tuple], k: int, uniref_id: str = None) -> object:
    return random.choices(array, weights, k=k)


def uniref_id_selector(uniref_id: str, procnum: int, worldsize: int) -> bool:
    return (simple_hash(uniref_id) + procnum) % worldsize == 0


def get_cluster_filename_for_rank(
    prefix: str,
    rank: int, worldsize: int,
    dtype: str = "uniref100"
) -> str:
    filenames = glob.glob(f"{prefix}*.pkl")
    if len(filenames) != worldsize:
        raise ValueError("Number of ranks needs to equal number of clusters")
    output_name = get_output_names(prefix, rank)[0] if dtype == "uniref100" else \
        f"{prefix}-{rank}.pkl"
    if output_name not in filenames:
        raise ValueError(f"Got prefix: {prefix}, output format: {filenames[0]}")
    return output_name


def process_records(
    fasta: str,
    max_sequence_length: int,
    max_masked_segment: int,
    include_extremities: bool,
    procnum: int,
    train_val_test_split: tuple,
    workdir: str,
    worldsize: int,
    max_splits_per_seq: int = -1,
    max_sequences_to_process: int = -1,
    min_masked_segment: int = 1,
    train_val_test_splitter: Optional[Callable] = None,
) -> None:
    if _TEST_RANDCHOICES_FUNCTOR is None:
        randchoices_functor = default_rand_choices_wrapper
    else:
        randchoices_functor = _TEST_RANDCHOICES_FUNCTOR

    if _TEST_RANDSAMPLE_FUNCTOR is None:
        randsample_functor = random.sample
    else:
        randsample_functor = _TEST_RANDSAMPLE_FUNCTOR

    if _TEST_RANDINT_FUNCTOR is None:
        randint_functor = random.randint
    else:
        randint_functor = _TEST_RANDINT_FUNCTOR

    if train_val_test_splitter is None:
        train_val_test_splitter = randchoices_functor

    tokenizer = data_processing.Tokenizer(
        max_sequence_length=max_sequence_length,
        max_masked_segment=max_masked_segment,
        include_extremities=include_extremities,
        randsample_functor=randsample_functor,
        randint_functor=randint_functor,
        silent=True,
        min_masked_segment=min_masked_segment,
    )

    counts = defaultdict(int)

    with open(os.path.join(
            workdir, f"assignments_proc_{procnum}.csv"), "w", newline="") as whandle:
        writer = csv.DictWriter(
            whandle, fieldnames=[
                "UniRefID", "fragment", "data_type", "encoder_seq", "decoder_seq"])
        writer.writeheader()

        fasta_reader = fasta_serial_reader(fasta)

        if max_sequences_to_process > 0:
            fasta_reader = islice(fasta_reader, 0, max_sequences_to_process)

        for fasta_item in fasta_reader:
            uniref_id = fasta_item.header

            # if (simple_hash(uniref_id) + procnum) % worldsize != 0:
            #     continue
            if not uniref_id_selector(uniref_id, procnum, worldsize):
                continue

            try:
                data_type = train_val_test_splitter(
                ["train", "val", "test"], weights=train_val_test_split, k=1, uniref_id=uniref_id)[0]
            except TrainValSplitter.ClusterNotFoundError:
                continue

            full_sequence = fasta_item.sequence

            all_splits = list(split_sequences(full_sequence, max_length=max_sequence_length))
            
            if max_splits_per_seq <= 0 or max_splits_per_seq >= len(all_splits):
                selected_splits = all_splits
            else:
                selected_splits = randsample_functor(all_splits, max_splits_per_seq)

            for i, sequence in enumerate(selected_splits):
                res = tokenizer.tokenize(sequence)
                if res is None:
                    continue
                encoder_seq, decoder_seq = res
                encoder_seq = json.dumps(encoder_seq)
                decoder_seq = json.dumps(decoder_seq)
                writer.writerow(
                    {"UniRefID": uniref_id, "fragment": i, "data_type": data_type,
                    "encoder_seq": encoder_seq, "decoder_seq": decoder_seq}
                )
                counts[data_type] += 1

        filename = whandle.name

    logger.info(f"Worker {procnum} completed")

    return counts, filename


def process_records_consolidator(results: list) -> tuple:
    all_counts = defaultdict(int)
    filenames_list = []

    for counts, filename in results:
        for key in counts:
            all_counts[key] += counts[key]
        filenames_list.append(filename)

    return all_counts, filenames_list


def num_segments_in_sequence(sequence: str, max_sequence_length: int) -> int:
    return math.ceil(len(sequence) / max_sequence_length)


def find_data_type(clusters: tuple, cluster_name: str) -> str:
    train_clusters, val_clusters, test_clusters = clusters
    clusters = {"train": train_clusters, "val": val_clusters, "test": test_clusters}
    for key in clusters:
        if cluster_name in clusters[key]:
            return key
    raise ValueError(f"Cannot find cluster {cluster_name} in {str(list(clusters.keys()))}")


def simple_hash(string: str) -> int:
    return sum(ord(x) for x in string)


def normalize(array: list) -> list:
    t = sum(array)
    return [i / t for i in array]


class SimpleSplitter:
    """
    Simple splitter which is already provided the dictionary mapping
    of train/val/test split
    """
    def __init__(self, dictionary: dict, *args, **kwargs):
        self.dictionary = dictionary
        self.scorecard = defaultdict(int)

    def __call__(self, array: list, weights: list, k: int, uniref_id: str) -> str:
        if uniref_id not in self.dictionary:
            raise TrainValSplitter.ClusterNotFoundError
        res = self.dictionary[uniref_id]
        self.scorecard[res] += 1
        return [res]


def setup_cluster_preprocessing(args: argparse.Namespace) -> Callable:
    global default_rand_choices_wrapper
    global uniref_id_selector

    if args.uniref100_cluster_data:
        cluster_data = args.uniref100_cluster_data
        splitter = TrainValSplitter
        dtype = "uniref100"
    else:
        cluster_data = args.uniref50_cluster_data
        splitter = SimpleSplitter
        dtype="uniref50"

    filename_for_rank = get_cluster_filename_for_rank(
        cluster_data, args.local_rank, args.worldsize, dtype=dtype)
    with open(filename_for_rank, "rb") as fhandle:
        dictionary = pickle.load(fhandle)
    train_val_test_splitter = splitter(
        dictionary, n_buckets=args.num_sampler_buckets)
    uniref_id_selector = lambda *args, **kwargs: True
    return train_val_test_splitter


def main(args: argparse.Namespace):
    train_val_test_splitter = None

    if args.uniref100_cluster_data or args.uniref50_cluster_data:
        logger.info("Setting up run for predefined clusters")
        train_val_test_splitter = setup_cluster_preprocessing(args)

    logger.info("Collecting train val test clusters")
    total_rows = 0
    train_val_test_split = normalize([float(x) for x in args.train_val_test_split.split(",")])

    tokenizer = data_processing.Tokenizer(
        max_sequence_length=args.max_sequence_length,
        max_masked_segment=args.max_masked_segment,
        include_extremities=args.include_extremities,
        min_masked_segment=args.min_masked_segment,
    )

    # Assign sequences to train/val/test groups and provide counts
    logger.info("Assigning sequences to train/val/test groups")
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    workdir = os.path.join(f"{args.datadir}_workdir_{timestamp}")
    if os.path.exists(workdir):
        raise ValueError(f"Workdir {workdir} exists, use a different prefix for datadir")
    os.makedirs(workdir)
    counts, filename = process_records(
        fasta=args.fasta,
        max_sequence_length=args.max_sequence_length,
        max_masked_segment=args.max_masked_segment,
        include_extremities=args.include_extremities,
        procnum=args.local_rank,
        train_val_test_split=train_val_test_split,
        workdir=workdir,
        worldsize=args.worldsize,
        max_splits_per_seq=args.max_splits_per_seq,
        max_sequences_to_process=args.max_sequences_to_process,
        min_masked_segment=args.min_masked_segment,
        train_val_test_splitter=train_val_test_splitter,
    )
    logger.info("Found the following sequence counts %s" % json.dumps(counts))

    # Reserve hard-disk space
    logger.info("Initializing datasets on disk")
    dsets = {
        key: data_processing.DiskStorage(
            max_length_encoder=tokenizer.max_encoder_length,
            max_length_decoder=tokenizer.max_decoder_length,
            num_sequences=counts[key],
            datadir=os.path.join(args.datadir, key),
            mode="w+",
        ) for key in counts.keys()
    }

    # Write datasets
    logger.info("Writing datasets")

    with tqdm.tqdm(total=sum(counts.values()), desc="Writing") as progressbar:
        with open(filename, "r", newline="") as fhandle:
            reader = csv.DictReader(fhandle)
            for row in reader:
                dsets[row["data_type"]].append(
                    json.loads(row["encoder_seq"]),
                    json.loads(row["decoder_seq"]),
                    json.dumps({"uniref_id": row["UniRefID"], "fragment": row["fragment"]})
                )
                progressbar.update(1)

    logger.info("Completed writes")
    shutil.rmtree(workdir)

    for key in dsets:
        dsets[key].tokenizer = copy.deepcopy(tokenizer)
        dsets[key].close()

    if args.uniref100_cluster_data or args.uniref50_cluster_data:
        logger.info(
            f"Completed runs (rank {args.local_rank}), sampling scorecard = {dict(train_val_test_splitter.scorecard)}")

    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create UniRef tokenized datasets")

    parser.add_argument(
        "--train_val_test_split", help="Train val test split", default="0.98,0.01,0.01",
    )

    parser.add_argument(
        "--max_sequence_length", help="Maximum sequence length", default=1022, type=int,
    )

    parser.add_argument(
        "--max_masked_segment", help="Maximum size of masked segment", default=63, type=int,
    )

    parser.add_argument(
        "--include_extremities",
        help="Include extremities in masked segment", default=False, action="store_true",
    )

    parser.add_argument(
        "--datadir", help="Data directory", required=True,
    )

    parser.add_argument(
        "--fasta", help="Fasta file", required=True)

    parser.add_argument(
        "--max_items_to_process", help="Process only a small number of items to prototype",
        type=int, required=False,
    )

    parser.add_argument(
        "--test", help="Run test", default=False, action="store_true",
    )

    parser.add_argument("--min_masked_segment", help="Minimum length to be masked",
        default=1, type=int)

    parser.add_argument(
        "--max_splits_per_seq",
        help="Maximum number of splits per sequence", type=int, default=-1)

    parser.add_argument(
        "--max_sequences_to_process",
        help="Only process the set number of sequences", type=int, default=-1)

    parser.add_argument(
        "--uniref100_cluster_data",
        help="Prefix of uniref100 clustering information",
        default=None,
    )

    parser.add_argument(
        "--uniref50_cluster_data",
        help="Predetermined UniRef50 train-val-test splits",
        default=None,
    )

    parser.add_argument(
        "--num_sampler_buckets",
        help="Number of buckets for train/val/test sampler for uniref100 clustering",
        type=int,
        default=10000,
    )

    parser.add_argument("--seed", help="Random seed", default=42, type=int)

    args = parser.parse_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.worldsize = int(os.environ["WORLD_SIZE"])
    args.datadir = os.path.join(args.datadir, str(args.local_rank))

    if args.test:
        _TEST_RANDCHOICES_FUNCTOR = RandChoicesFunctor()
        _TEST_RANDINT_FUNCTOR = RandIntFunctor()
        _TEST_RANDSAMPLE_FUNCTOR = RandSampleFunctor()

    logging.basicConfig(
        level=(logging.INFO if args.local_rank == 0 else logging.WARNING),
        format="%(asctime)s %(levelname)s %(message)s")

    random.seed(args.seed)

    main(args)
