# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import train_prot_gpt2
from transformers import AutoTokenizer
import shutil
import os
import torch


def test_dataset():
    tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    testdir = "/tmp/test_protgpt2/test_dataset"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    with open(os.path.join(testdir, "somefile.fa"), "w") as fhandle:
        fhandle.write(f">1\nABCDEFG*\n")
        fhandle.write(f">2\nGTA*\n")
        fname = fhandle.name

    dataset = train_prot_gpt2.Dataset(fname)

    res0 = [0] + tokenizer("ABCDEFG*")["input_ids"]
    res1 = [0] + tokenizer("GTA*")["input_ids"]

    assert(dataset[0] == res0), f"{dataset[0]} != {res0}"
    assert(dataset[1] == res1)

    res = train_prot_gpt2.collate_function([dataset[0], dataset[1]])

    assert(len(res0) > len(res1)), "Check the test again, tokenization lengths are different"

    exp_input_ids = torch.LongTensor(
            [
                res0,
                res1 + [0] * (len(res0) - len(res1)),
            ]
        )
    exp_attention_mask = torch.ByteTensor(
            [
                [
                    [1] * len(res0),
                    [1] * len(res1) + [0] * (len(res0) - len(res1))
                ]
            ]
        )

    def tensor_eq(a, b):
        return torch.all(a == b)

    assert(tensor_eq(exp_input_ids, res["input_ids"]))
    assert(tensor_eq(exp_attention_mask, res["attention_mask"]))

    print("Test test_dataset passed")


if __name__ == "__main__":
    test_dataset()

