# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import torch
import torch.distributed
import torch.utils.data
from transformers import HfArgumentParser, TrainingArguments
import sys
import quark_finetune
from functools import reduce
import random


_TENSOR_DICT = [
    {
        "input_ids": torch.LongTensor([26, 2, 3, 4, 5, 6, 7, 8, 9]),
        "attention_mask": torch.ByteTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "quantile_token": 13,
    },
    {
        "input_ids": torch.LongTensor([26, 4, 1, 2, 3, 9, 0]),
        "attention_mask": torch.ByteTensor([1, 1, 1, 1, 1, 1, 0]),
        "quantile_token": 32,
    },
    {
        "input_ids": torch.LongTensor([26, 7, 9, 0]),
        "attention_mask": torch.ByteTensor([1, 1, 1, 0]),
        "quantile_token": 34,
    },
    {
        "input_ids": torch.LongTensor([26, 1, 9, 3, 4]),
        "attention_mask": torch.ByteTensor([1, 1, 1, 0, 0]),
        "quantile_token": 33,
    }
]


def tensor_eq(x, y):
    convert = lambda a: {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in a.items()}
    x = [convert(i) for i in x]
    y = [convert(j) for j in y]
    return x == y


class CustomSampler:
    def __init__(self):
        pass

    def __call__(self, array, sample_size):
        res = array[-2: ]
        random.shuffle(res)
        return res


def get_index(collected):
    train_seq, ref_seq = collected
    train_seq = {key: value.tolist()[0] for key, value in train_seq.items()}
    ref_seq = {key: value.tolist()[0] for key, value in ref_seq.items()}
    assert(train_seq['input_ids'][0] == ref_seq['input_ids'][0] and 
        train_seq['input_ids'][2:] == ref_seq['input_ids'][1:]
    )
    assert(train_seq['attention_mask'][0] == ref_seq['attention_mask'][0] and 
        train_seq['attention_mask'][2:] == ref_seq['attention_mask'][1:]
    )
    def compare(ref_seq, tensor_dict):
        tensor_dict_list = tensor_dict["input_ids"].masked_select(
            tensor_dict["attention_mask"] == 1).tolist()
        ref_seq_list = ref_seq["input_ids"][: sum(ref_seq["attention_mask"])]
        return ref_seq_list == tensor_dict_list
    res = [i for i, x in enumerate(_TENSOR_DICT) if compare(ref_seq, x)]
    return res.pop()


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ))
    training_args, = parser.parse_args_into_dataclasses()

    if training_args.world_size != 2:
        raise sys.exit("Please launch the test with world_size=2")

    """ Seed the processes differently so that the CustomSampler orders them differently"""
    random.seed(training_args.process_index + 13)

    """ Collect _TESOR_DICT[0: 2] from the two processes and verify """
    """ This tests the all_gather function in init_training and evaluation_loop """
    results = quark_finetune.all_gather_list_torchrun(
        training_args,
        list_to_send=_TENSOR_DICT[training_args.process_index: training_args.process_index + 1]
    )
    assert(tensor_eq(results, _TENSOR_DICT[0: 2])), f"{results} != {_TENSOR_DICT[0: 2]}"

    """ Create data pool """
    pool = quark_finetune.DataPool(
        pool=results,
        eos_token=9,
        fixed_epoch_length=2,
        max_length=15,
        randsampler=CustomSampler(),
    )

    """ Create DataLoader with DistributedSampler just like in huggingface Trainer """
    dataloader = torch.utils.data.DataLoader(
        pool,
        batch_size=1,
        sampler=torch.utils.data.DistributedSampler(
            pool,
            num_replicas=training_args.world_size,
            rank=training_args.process_index,
            seed=training_args.seed,
            shuffle=True,
        ),
        collate_fn=quark_finetune.collate_function,
    )
    collected_indices = []

    """ Perform set_epoch similar to the case in Trainer """
    dataloader.sampler.set_epoch(0)

    """ Iterate through dataloader, and identify the index number of the collected item
    in each process. Then consolidate and check that it's the first two items in
    _TENSOR_DICT that we received. This checks the logic in init_training """
    for i, item in enumerate(dataloader):
        collected_indices.append(get_index(item))

    all_collected_indices = [None, None]
    torch.distributed.all_gather_object(all_collected_indices, collected_indices)

    all_collected_indices = reduce(lambda a, b: a + b, all_collected_indices, [])
    assert(set(all_collected_indices) == {0, 1})

    """ Next, pivot to the evaluation_loop. This time, we assume generation produces 
    items [2, 3] from _TENSOR_DICT """
    results = quark_finetune.all_gather_list_torchrun(
        training_args,
        list_to_send=_TENSOR_DICT[2: 3] if training_args.process_index == 0 else _TENSOR_DICT[3: 4]
    )
    assert(tensor_eq(results, _TENSOR_DICT[2: 4])), f"{results} != {_TENSOR_DICT[2: 4]}"

    """ We add the items to the pool, sample, and broadcast the indices to all processes,
    so that all processes have the same sample ordering """
    pool.extend(results)
    pool.sample_epoch()
    torch.distributed.broadcast_object_list(pool.subset_indices, src=0)

    """ Next, we run DataLoader again and we verify that the collected batches across
    all processes correspond to items [2, 3] in _TENSOR_DICT """
    collected_indices = []

    dataloader.sampler.set_epoch(1)

    for i, item in enumerate(dataloader):
        collected_indices.append(get_index(item))

    all_collected_indices = [None, None]
    torch.distributed.all_gather_object(all_collected_indices, collected_indices)

    all_collected_indices = reduce(lambda a, b: a + b, all_collected_indices, [])
    assert(set(all_collected_indices) == {2, 3}), f"{all_collected_indices} != [2, 3]"

    print("Distributed synchronization test passed")
