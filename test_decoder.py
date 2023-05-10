# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import data_processing
import torch
import random_split_fasta
import pandas
from argparse import Namespace
import os
import shutil
import pysam
from collections import defaultdict
from competition_models import DecoderPooler


def test_data():
    testdir = "/tmp/test_decoder/test_data"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    seq_list = [
        "ABCDEFG",
        "DEFG",
        "XYZABCACACXAG",
    ]

    with open(os.path.join(testdir, "test_fa.fa"), "w") as fhandle:
        for i, seq in enumerate(seq_list):
            fhandle.write(f">{i}\n{seq}\n")
        seq_file = fhandle.name

    data = data_processing.SimpleSequenceDataset(
        seq_file=seq_file,
        max_length=9,
        randint_functor=lambda *args, **kwargs: 2,
    )

    assert(len(data) == 3)
    collated = data_processing.collate_function_for_decoder([data[0], data[1], data[2]])
    mapper = data_processing.Tokenizer().mapper

    seq0 = [mapper[i] for i in ["[CLS]"] + list("ABCDEFG")]
    seq1 = [mapper[i] for i in ["[CLS]"] + list("DEFG")]
    seq2 = [mapper[i] for i in ["[CLS]"] + list("ZABCACAC")]

    assert(list(collated["input_ids"].shape) == [3, 9])
    assert(list(collated["attention_mask"].shape) == [3, 9])
    assert(list(collated["labels"].shape) == [3, 9])

    exp_input_ids = torch.LongTensor([
        seq0 + [0] * 1,
        seq1 + [0] * 4,
        seq2,
    ])

    exp_attention_mask = torch.ByteTensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    exp_labels = torch.LongTensor([
        seq0 + [-100] * 1,
        seq1 + [-100] * 4,
        seq2,
    ])

    def tensor_compare(a, b):
        return torch.all(a == b)

    assert(tensor_compare(exp_input_ids, collated["input_ids"])), f"%s != %s" % (str(exp_input_ids), str(collated["input_ids"]))
    assert(tensor_compare(exp_attention_mask, collated["attention_mask"]))
    assert(tensor_compare(exp_labels, collated["labels"]))
    assert(list(collated.keys()) == ["input_ids", "attention_mask", "labels"])

    print("Test test_data passed")


def test_data_ignore_too_long():
    testdir = "/tmp/test_decoder/test_data_ignore_too_long"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    seq_list = [
        "ABCDEFG",
        "DEFG",
        "XYZABCACACXAG",
    ]

    with open(os.path.join(testdir, "test_fa_ignore_too_long.fa"), "w") as fhandle:
        for i, seq in enumerate(seq_list):
            fhandle.write(f">{i}\n{seq}\n")
        seq_file = fhandle.name

    data = data_processing.SimpleSequenceDataset(
        seq_file=seq_file,
        max_length=9,
        randint_functor=lambda *args, **kwargs: 2,
        ignore_too_long=True,
    )

    assert(len(data) == 2)
    collated = data_processing.collate_function_for_decoder([data[0], data[1]])
    mapper = data_processing.Tokenizer().mapper

    seq0 = [mapper[i] for i in ["[CLS]"] + list("ABCDEFG")]
    seq1 = [mapper[i] for i in ["[CLS]"] + list("DEFG")]

    assert(list(collated["input_ids"].shape) == [2, 9])
    assert(list(collated["attention_mask"].shape) == [2, 9])
    assert(list(collated["labels"].shape) == [2, 9])

    exp_input_ids = torch.LongTensor([
        seq0 + [0] * 1,
        seq1 + [0] * 4,
    ])

    exp_attention_mask = torch.ByteTensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
        ]
    )

    exp_labels = torch.LongTensor([
        seq0 + [-100] * 1,
        seq1 + [-100] * 4,
    ])

    def tensor_compare(a, b):
        return torch.all(a == b)

    assert(tensor_compare(exp_input_ids, collated["input_ids"])), f"%s != %s" % (str(exp_input_ids), str(collated["input_ids"]))
    assert(tensor_compare(exp_attention_mask, collated["attention_mask"]))
    assert(tensor_compare(exp_labels, collated["labels"]))
    assert(list(collated.keys()) == ["input_ids", "attention_mask", "labels"])

    print("Test test_data_ignore_too_long passed")


def test_random_split_fasta(enumerate: bool = False):
    testdir = os.path.join("/tmp/test_random_split_fasta")

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    ref = "XACGT"

    df = pandas.DataFrame.from_dict(
        {
            "AA Substitutions": [
                "Spike_A2B",
                "Spike_C3A",
                "Spike_C3A",
                "Spike_G4del",
                "Spike_T5A",
                "Spike_G4del",
                "Spike_C3A",
                "Spike_G4del",
                "Spike_A2B",
                "Spike_X1Z",
                "Spike_X1Z",
                "Spike_X1P",
                "Spike_X1F",
            ],
            "Submission date": [
                "2021-01-01",
                "2021-01-03",
                "2021-01-03",
                "2021-01-05",
                "2021-01-05",
                "2021-01-07",
                "2021-01-08",
                "2021-01-09",
                "2021-01-10",
                "2021-01-09",
                "2021-01-09",
                "2021-01-09",
                "2021-01-10",
            ],
            "Pango lineage": ["A", "B", "B", "C", "D", "C", "B", "C", "A", "E", "E", "E", "F"],
            "Location": ["Europe"] * 13,
            "Host": ["human"] * 13,
            "reconstruction_success_Spike": [True] * 13,
            "Accession ID": list(range(13)),
        }
    )
    """
    Lineage counts:
    A: 1, (one of them is not included)
    B: 3,
    C: 3,
    D: 1,
    E: 3,
    F: 1 (not included)

    In order, that is
    C, E, B: B, C, E (sorted)
    A, D   : A, D (sorted)

    Train: B, A
    Val : C, D
    Test: E

    Train sequences: [C3A, A2B]
    Val sequences: [G4del, T5A]
    Test: [X1Z, X1P]
    """

    variants_name = os.path.join(testdir, "variants.tsv")
    df.to_csv(variants_name, sep="\t")

    with open(os.path.join(testdir, "ref.txt"), "w") as fhandle:
        ref_filename = fhandle.name
        fhandle.write(ref)

    output_prefix = os.path.join(testdir, "output_prefix")

    args = Namespace(
        prefix=output_prefix,
        tsv=variants_name,
        last_date="2021-01-09",
        ref=ref_filename,
        n_train_per_bucket=1,
        n_val_per_bucket=1,
        n_test_per_bucket=1,
        protein="Spike",
        datefield="Submission date",
        enumerate=enumerate,
        randshuffler=lambda x: x.sort(),
    )
    
    random_split_fasta.main(args)

    def get_sequence_counts(filename: str) -> dict:
        counts_dict = defaultdict(int)

        with pysam.FastaFile(filename) as fhandle:
            for r in fhandle.references:
                counts_dict[fhandle.fetch(r)] += 1

        return dict(counts_dict)

    def get_mutation_sequences(filename: str) -> dict:
        with open(filename, "r") as fhandle:
            sequences = set(x.strip() for x in fhandle)
        return sequences

    expected_counts = {
        "XAAGT": 3,
        "XBCGT": 1,
    }

    expected_val_counts = {
        "XACT": 3,
        "XACGA": 1,
    }
    
    expected_test_counts = {
        "ZACGT": 2,
        "PACGT": 1,
    }

    if not enumerate:
        expected_counts = {x: 1 for x in expected_counts}
        expected_val_counts = {x: 1 for x in expected_val_counts}
        expected_test_counts = {x: 1 for x in expected_test_counts}

    obtained_train_counts = get_sequence_counts(f"{output_prefix}.train.fa")
    obtained_val_counts = get_sequence_counts(f"{output_prefix}.val.fa")
    obtained_test_counts = get_sequence_counts(f"{output_prefix}.test.fa")

    assert(expected_counts == obtained_train_counts), f"{expected_counts} != {obtained_train_counts}"
    assert(expected_val_counts == obtained_val_counts), f"{expected_val_counts} != {obtained_val_counts}"
    assert(expected_test_counts == obtained_test_counts), f"{expected_test_counts} != {obtained_test_counts}"

    obtained_train_muts = get_mutation_sequences(f"{output_prefix}.train.mutations.lst")
    obtained_val_muts = get_mutation_sequences(f"{output_prefix}.val.mutations.lst")
    obtained_test_muts = get_mutation_sequences(f"{output_prefix}.test.mutations.lst")

    assert(obtained_train_muts == {"A2B", "C3A"})
    assert(obtained_val_muts == {"G4del", "T5A"})
    assert(obtained_test_muts == {"X1Z", "X1P"})

    print(f"Test test_random_split_fasta (enumerate={enumerate}) passed")


def test_calc_likelihoods():
    logits = torch.Tensor([
        [
            [1, 2, -1], [3, 4, -2], [5, 6, 9], [7, 8, 2], [9, 10, 0],
        ]
    ])
    input_ids = torch.LongTensor([
        [2, 0, 1, 0, 0]
    ])
    attention_mask = torch.ByteTensor([
        [1, 1, 1, 1, 0]
    ])
    ll_calc = calc_likelihoods(logits, {"input_ids": input_ids, "attention_mask": attention_mask})

    all_lls = torch.log_softmax(logits, dim=-1)
    ll_exp = (all_lls[0, 0, 0] + all_lls[0, 1, 1] + all_lls[0, 2, 0]).item()
    assert(ll_exp - 1e-5 <= ll_calc.item() <= ll_exp + 1e-5)
    print("Test test_calc_likelihoods passed")


def test_pooler():
    pooler = DecoderPooler()
    last_layer_output = torch.randn(3, 4, 2)
    attention_mask = torch.Tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]).byte()
    expected_output = torch.stack([
        last_layer_output[0, 2],
        last_layer_output[1, 1],
        last_layer_output[2, 3]
    ], dim=0)
    obtained_result = pooler(last_layer_output, attention_mask)
    assert(torch.all(obtained_result == expected_output))
    print("Test test_pooler passed")


if __name__ == "__main__":
    test_data()
    test_data_ignore_too_long()
    test_random_split_fasta(enumerate=False)
    test_random_split_fasta(enumerate=True)
    # test_calc_likelihoods()
    test_pooler()    
