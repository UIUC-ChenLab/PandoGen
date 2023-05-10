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
from typing import Callable, Generator, Union
from collections.abc import Iterable
import random
import copy
import datetime
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()


_TEST_RANDSAMPLE_FUNCTOR = None
_TEST_RANDINT_FUNCTOR = None
_TEST_RANDCHOICES_FUNCTOR = None


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


def process_records(
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    fasta: str,
    max_sequence_length: int,
    max_masked_segment: int,
    include_extremities: bool,
    procnum: int,
    train_val_test_split: tuple,
    workdir: str,
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

    tokenizer = data_processing.Tokenizer(
        max_sequence_length=max_sequence_length,
        max_masked_segment=max_masked_segment,
        include_extremities=include_extremities,
        randsample_functor=randsample_functor,
        randint_functor=randint_functor,
        silent=True,
    )

    counts = defaultdict(int)
    filename = None

    with pysam.FastaFile(fasta) as fhandle, \
        open(os.path.join(
            workdir, f"assignments_proc_{procnum}.csv"), "w", newline="") as whandle:
        writer = csv.DictWriter(
            whandle, fieldnames=[
                "UniRefID", "fragment", "data_type", "encoder_seq", "decoder_seq"])
        writer.writeheader()

        while True:
            work_item = input_queue.get()
            if work_item == "PROCESS COMPLETED":
                logger.info(f"Worker {procnum} received termination signal")
                break
            uniref_id = work_item
            data_type = randchoices_functor(
                ["train", "val", "test"], weights=train_val_test_split, k=1, uniref_id=uniref_id)[0]
            full_sequence = fhandle.fetch(uniref_id)
            for i, sequence in enumerate(
                split_sequences(full_sequence, max_length=max_sequence_length)):
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

    output_queue.put((counts, filename))


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


def parallel_scheduler(
    functor: Callable,
    num_workers: int,
    init_kwargs: dict,
    argstream: Iterable,
    collect_returns: bool = False,
    desc: str = "Processing",
    total: int = None,
    consolidator: Callable = lambda x: x,
):
    """
    Common interface for multiprocessing
    """
    results = None
    total_args = 0

    # Initialize arguments
    send_queue = multiprocessing.Queue(maxsize=128)
    receive_queue = multiprocessing.Queue()

    # Initialize processes
    processes = []

    for i in range(num_workers):
        kwargs = copy.deepcopy(init_kwargs)
        kwargs["input_queue"] = send_queue
        if collect_returns:
            kwargs["output_queue"] = receive_queue
        kwargs["procnum"] = i
        processes.append(multiprocessing.Process(target=functor, kwargs=kwargs))

    # Start processes    
    for p in processes:
        p.start()

    # Send arguments in iterable
    for arg in tqdm.tqdm(argstream, desc=desc, total=total):
        send_queue.put(arg)
        total_args += 1

    # Indicate that last arguments have been sent
    for p in processes:
        send_queue.put("PROCESS COMPLETED")

    # Collect results if any
    if collect_returns:
        results = [receive_queue.get() for p in processes]

    # Terminate processes
    for p in processes:
        p.join()

    return consolidator(results), total_args


def fasta_id_dispenser(fasta_file: str) -> Generator[str, None, None]:
    with pysam.FastaFile(fasta_file) as fhandle:
        for r in fhandle.references:
            yield r


def normalize(array: list) -> list:
    t = sum(array)
    return [i / t for i in array]


def main(args):
    logger.info("Collecting train val test clusters")
    total_rows = 0
    train_val_test_split = normalize([float(x) for x in args.train_val_test_split.split(",")])

    tokenizer = data_processing.Tokenizer(
        max_sequence_length=args.max_sequence_length,
        max_masked_segment=args.max_masked_segment,
        include_extremities=args.include_extremities,
    )

    # Assign sequences to train/val/test groups and provide counts
    logger.info("Assigning sequences to train/val/test groups")
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    workdir = os.path.join(f"{args.datadir}_workdir_{timestamp}")
    if os.path.exists(workdir):
        raise ValueError(f"Workdir {workdir} exists, use a different prefix for datadir")
    os.makedirs(workdir)
    results, total_args = parallel_scheduler(
        functor=process_records,
        num_workers=max(1, args.num_workers),
        init_kwargs={
            "fasta": args.fasta,
            "max_sequence_length": args.max_sequence_length,
            "train_val_test_split": train_val_test_split,
            "max_masked_segment": args.max_masked_segment,
            "include_extremities": args.include_extremities,
            "workdir": workdir,
        },
        argstream=fasta_id_dispenser(args.fasta),
        collect_returns=True,
        desc="Counting number of examples",
        total=None,
        consolidator=process_records_consolidator,
    )
    counts, filenames = results
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
        for filename in filenames:
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
        "--num_workers", help="Number of CPU processes to use", default=0, type=int,
    )

    parser.add_argument("--seed", help="Random seed", default=42, type=int)

    args = parser.parse_args()

    random.seed(args.seed)

    main(args)
