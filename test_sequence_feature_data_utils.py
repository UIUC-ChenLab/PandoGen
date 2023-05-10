# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import sequence_feature_data_utils
import torch
import os
import numpy as np
import shutil
import pandas
import datetime
import data_processing
import train_competition
import create_occurrence_buckets
from argparse import Namespace
import copy
import logging
import itertools
import warnings

warnings.filterwarnings("error")

create_occurrence_buckets._MIN_OCCURRENCE_MAX_SEQUENCE = 0


def test_create_embeddings_data(testdir: str):
    data = sequence_feature_data_utils.EmbeddingsMemmapData(
        os.path.join(testdir, "embeddings"), n_items=2, max_length=4, embedding_dim=3,
    )
    n501y = (np.array([
            [0, 1, 2],
            [1, 2, 1],
            [-1, -1, -2],
            [-4, -1, -2],
        ]),
        np.array(
            [1, 1, 0, 0],
            dtype=np.uint8,
        )
    )
    data["N501Y"] = n501y
    d651a = (
        np.array([
            [3, 4, 1],
            [-1, -1, -2],
            [-3, -4, -5],
            [-7, -8, -10],
        ]),
        np.array([1, 0, 0, 0]),
    )
    data["D651A"] = d651a
    data.close()
    vdata = sequence_feature_data_utils.EmbeddingsMemmapData(os.path.join(testdir, "embeddings"))
    assert(len(vdata) == 2)

    def cmp(a, b):
        a0 = torch.Tensor(a[0])
        a1 = torch.ByteTensor(a[1])
        assert(torch.all(a0 == b[0]))
        assert(torch.all(a1 == b[1]))

    cmp(d651a, vdata["D651A"])
    cmp(n501y, vdata["N501Y"])
    print("Test test_create_embeddings_data passed")


def test_pairwise_dataset():
    ref = "XACGT"

    df = pandas.DataFrame.from_dict(
        {
            "SpikeMutations": ["A2B", "C3A", "C3A", "G4del", "T5A", "G4del"],
            "Submitted Date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-03", "2021-01-04", "2021-01-04"],
        }
    )
    df = df.assign(ParsedDate=pandas.to_datetime(df["Submitted Date"]))
    d = discovery_end_date=datetime.datetime.strptime("2021-01-06", "%Y-%m-%d")
    sequence_buckets, all_sequences, sequence_counts = sequence_feature_data_utils.prepare_pariwise_data_preliminaries(
        df,
        discovery_end_date=d,
        availability_end_date=d,
        period_length=2,
    )
    data = sequence_feature_data_utils.PairwiseDataset(
        sequence_buckets,
        all_sequences,
        sequence_counts,
        ref=ref,
        randint_functor=lambda *args, **kwargs: 0,
        randshuffle_functor=lambda *args, **kwargs: 0,
    )
    sequences = [
        ["[CLS]"] + list("XBCGT") + ["[SEP]"],
        ["[CLS]"] + list("XAAGT") + ["[SEP]"],
        ["[CLS]"] + list("XACT") + ["[SEP]"],
        ["[CLS]"] + list("XACGA") + ["[SEP]"],
    ]
    tokenizer = data_processing.Tokenizer()
    tokens = [
        [tokenizer.mapper[x] for x in seq] for seq in sequences
    ]

    def frac_cmp(a, b, eps=1e-4):
        return(a - eps <= b <= a + eps)

    assert(len(data) == 4)
    res0 = data[0]
    (tokens0, tokens1), frac = res0
    assert(tokens0.tolist() == tokens[0]), f"{tokens0}!={tokens[0]}"
    assert(tokens1.tolist() == tokens[1])
    assert(frac_cmp(frac, 2/3))

    res1 = data[1]
    (tokens0, tokens1), frac = res1
    assert(tokens0.tolist() == tokens[1])
    assert(tokens1.tolist() == tokens[0])
    assert(frac_cmp(frac, 1/3))

    res2 = data[2]
    (tokens0, tokens1), frac = res2
    assert(tokens0.tolist() == tokens[2])
    assert(tokens1.tolist() == tokens[3])
    assert(frac_cmp(frac, 1/3))

    res3 = data[3]
    (tokens0, tokens1), frac = res3
    assert(tokens0.tolist() == tokens[3])
    assert(tokens1.tolist() == tokens[2])
    assert(frac_cmp(frac, 2/3))

    collated = sequence_feature_data_utils.collate_function(
        [data[0], data[1], data[2], data[3]],        
    )

    assert(len(collated) == 3)

    def compare(tensors, tokens):
        input_ids = tensors["input_ids"]
        attention_mask = tensors["attention_mask"]

        for i, (a, b) in enumerate(zip(
            torch.unbind(input_ids, dim=0),
            torch.unbind(attention_mask, dim=0),
        )):
            selected = a.masked_select(b == 1).tolist()
            assert(selected == tokens[i]), f"{selected} != {tokens[i]}, index={i}"

    compare(collated[0], tokens)
    compare(collated[1], [tokens[1], tokens[0], tokens[3], tokens[2]])

    for a, b in zip(collated[2], [2/3, 1/3, 1/3, 2/3]):
        assert(frac_cmp(a, b))

    print("Test test_pairwise_dataset passed")


def test_pairwise_dataset_embeddings():
    ref = "XACGT"

    testdir = "/tmp/test_pairwise_dataset_embeddings"

    embeddings = sequence_feature_data_utils.EmbeddingsMemmapData(
        os.path.join(testdir, "embeddings"), n_items=4, max_length=4, embedding_dim=3,
    )
    a2b = (np.array([
            [0, 1, 2],
            [1, 2, 1],
            [-1, -1, -2],
            [-4, -1, -2],
        ]),
        np.array([1, 1, 0, 0], dtype=np.uint8)
    )
    embeddings["A2B"] = a2b
    c3a = (
        np.array([
            [3, 4, 1],
            [-1, -1, -2],
            [-3, -4, -5],
            [-7, -8, -10],
        ]),
        np.array([1, 0, 0, 0], dtype=np.uint8),
    )
    embeddings["C3A"] = c3a
    g4x = (
        np.array([
            [3, 2, 2],
            [-1, -1, -2],
            [-3, -4, -5],
            [-7, -8, -10],
        ]),
        np.array([1, 0, 0, 0], dtype=np.uint8),
    )
    embeddings["G4X"] = g4x
    t5a = (
        np.array([
            [0, 1, 2],
            [1, 3, -1],
            [-1, -1, -2],
            [-4, -1, -2],
        ]),
        np.array([1, 1, 0, 0], dtype=np.uint8)
    )
    embeddings["T5A"] = t5a
    embeddings.close()
    embeddings = sequence_feature_data_utils.EmbeddingsMemmapData(
        os.path.join(testdir, "embeddings"))
    df = pandas.DataFrame.from_dict(
        {
            "SpikeMutations": ["A2B", "C3A", "C3A", "G4X", "T5A", "G4X"],
            "Submitted Date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-03", "2021-01-04", "2021-01-04"],
        }
    )
    df = df.assign(ParsedDate=pandas.to_datetime(df["Submitted Date"]))
    d = discovery_end_date=datetime.datetime.strptime("2021-01-06", "%Y-%m-%d")
    sequence_buckets, all_sequences, sequence_counts = sequence_feature_data_utils.prepare_pariwise_data_preliminaries(
        df,
        discovery_end_date=d,
        availability_end_date=d,
        period_length=2,
    )
    data = sequence_feature_data_utils.PairwiseDataset(
        sequence_buckets,
        all_sequences,
        sequence_counts,
        ref=ref,
        randint_functor=lambda *args, **kwargs: 0,
        randshuffle_functor=lambda *args, **kwargs: 0,
        embeddings=embeddings,
    )

    def frac_cmp(a, b, eps=1e-4):
        return(a - eps <= b <= a + eps)

    def compare_items(a, b):
        a = tuple(x.tolist() for x in a)
        b = tuple(x.tolist() for x in b)
        return(a == b)

    assert(len(data) == 4)
    res0, frac = data[0]
    assert(compare_items(res0[0], a2b) and compare_items(res0[1], c3a))
    assert(frac_cmp(frac, 2/3))

    res1, frac = data[1]
    assert(compare_items(res1[1], a2b) and compare_items(res1[0], c3a))
    assert(frac_cmp(frac, 1/3))

    res2, frac = data[2]
    assert(compare_items(res2[0], g4x) and compare_items(res2[1], t5a))
    assert(frac_cmp(frac, 1/3))

    res3, frac = data[3]
    assert(compare_items(res3[1], g4x) and compare_items(res3[0], t5a))
    assert(frac_cmp(frac, 2/3))

    collated = sequence_feature_data_utils.collate_function(
        [data[0], data[1], data[2], data[3]]
    )

    assert(len(collated) == 3)

    def compare(input_ids, attention_mask, list_of_tensors):
        for i, (a, b) in enumerate(
            zip(torch.unbind(input_ids, dim=0), torch.unbind(attention_mask, dim=0))):
            c, d = list_of_tensors[i]
            assert(torch.all(a == torch.Tensor(c)))
            assert(torch.all(b == torch.Tensor(d).byte()))
    
    compare(*collated[0], [a2b, c3a, g4x, t5a])
    compare(*collated[1], [c3a, a2b, t5a, g4x])

    for a, b in zip(collated[2], [2/3, 1/3, 1/3, 2/3]):
        assert(frac_cmp(a, b))

    shutil.rmtree(testdir)

    print("Test test_pairwise_dataset_embeddings passed")


def test_pairwise_bucketized_dataset():
    ref = "XACGT*"

    testdir = "/tmp/test_pairwise_bucketized_dataset"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    """
    We want train sequences to be [A2B, G4del] and
    val sequences to be [C3A, T5A]
    Counts are:
    A2B = 2
    G4del = 3
    C3A = 3
    T5A = 1

    Sorting in descending order gives us
    ['C3A', 'G4del', 'A2B', 'X1Z', 'T5A', 'X1F'] (verified this order from the output of sort)

    On the training side ...

    Training Initial: [C3A, A2B, T5A]
    Val initial: [G4del, X1Z, X1F]

    Training combinations:
    C3A -> A2B
        C3A has a lead time of 8, A2B has 10. This is accepted as the ratio is 1/3

    C3A -> T5A
        C3A has a lead time of 8, T5A has a lead time of 6. This is accepted as the ratio is 1 / 3

    A2B -> T5A
        T5A has a lead time of 6, A2B has a lead time of 10. Rejected as ratio is 1

    Validation combinations:
    G4del -> X1Z
        G4del has a lead time of 6, X1Z has a lead time of 2
        G4del has occurrence of 1 in a window of size 2, and X1Z has occurrence 2. However,
        given the tolerance between them = 4, both of them violate the minimum lead time
        requirement.

    G4del -> X1F
        Same as above

    X1F -> X1Z
        Same as above
    """
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
                "Spike_X1F",
                # Sequences reported after 2021-01-10 are not counted
                "Spike_C3A",
                "Spike_C3A",
                "Spike_C3A",
                "Spike_C3A",
                "Spike_C3A",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_A2B",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_T5A",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
                "Spike_X1F",
                "Spike_X1Z",
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
                "2021-01-10",
                # Sequences reported after 2021-01-10 are not considered
                "2021-01-11",
                "2021-01-12",
                "2021-01-13",
                "2021-01-14",
                "2021-01-15",
                "2021-01-16",
                "2021-01-17",
                "2021-01-18",
                "2021-01-19",
                "2021-01-20",
                "2021-01-21",
                "2021-01-22",
                "2021-01-23",
                "2021-01-24",
                "2021-01-25",
                "2021-01-26",
                "2021-01-27",
                "2021-01-28",
                "2021-01-29",
                "2021-01-30",
                "2021-01-31",
                "2021-02-01",
                "2021-02-02",
                "2021-02-03",
                "2021-02-04",
                "2021-02-05",
                "2021-02-06",
                "2021-02-07",
                "2021-02-08",
                "2021-02-09",
                "2021-02-10",
                "2021-02-11",
                "2021-02-12",
                "2021-02-13",
                "2021-02-14",
                "2021-02-15",
                "2021-02-16",
                "2021-02-17",
                "2021-02-18",
                "2021-02-19",
                "2021-02-20",
                "2021-02-21",
                "2021-02-22",
                "2021-02-23",
                "2021-02-24",
                "2021-02-25",
                "2021-02-26",
                "2021-02-27",
                "2021-02-28",
                "2021-03-01",
                "2021-03-02",
                "2021-03-03",
                "2021-03-04",
                "2021-03-05",
                "2021-03-06",
                "2021-03-07",
                "2021-03-08",
                "2021-03-09",
                "2021-03-10",
                "2021-03-11",
                "2021-03-12",
                "2021-03-13",
                "2021-03-14",
                "2021-03-15",
                "2021-03-16",
                "2021-03-17",
                "2021-03-18",
                "2021-03-19",
                "2021-03-20",
                "2021-03-21",
                "2021-03-22",
                "2021-03-23",
                "2021-03-24",
                "2021-03-25",
                "2021-03-26",
                "2021-03-27",
                "2021-03-28",
                "2021-03-29",
                "2021-03-30",
                "2021-03-31",
                "2021-04-01",
                "2021-04-02",
                "2021-04-03",
                "2021-04-04",
                "2021-04-05",
                "2021-04-06",
                "2021-04-07",
                "2021-04-08",
                "2021-04-08",
            ],
            "Pango lineage": ["A"] * (12 + 89),
            "Location": ["Europe"] * (12 + 89),
            "Host": ["human"] * (12 + 89),
            "reconstruction_success_Spike": [True] * (12 + 89),
            "Accession ID": list(range(12 + 89)),
        }
    )
    tsv = os.path.join(testdir, "test.tsv")
    df.to_csv(tsv, sep="\t")

    create_occurrence_buckets._FRACTIONS = [
        create_occurrence_buckets.CompetitionBrackets(0, 0.6, 5),
        create_occurrence_buckets.CompetitionBrackets(0.6, 1, 0),
    ]
    create_occurrence_buckets._MIN_LEAD_TIME = 0

    results = create_occurrence_buckets.get_training_pairs_from_tsv(
        tsv,
        availability_last_date="2021-01-10",
        train_seqs_per_bucket=1,
        val_seqs_per_bucket=1,
        period_length=1,
        randshuffle_functor=lambda *args, **kwargs: args[0],
    )
    train_pairings, val_pairings, test_pairings = results

    assert(not test_pairings)
    assert(not val_pairings)
    assert(len(train_pairings) == 2)

    train_pairings_expected = [
        create_occurrence_buckets.SeqPair(seq0='C3A', count0=3.0, seq1='A2B', count1=1.0, period=8),
        create_occurrence_buckets.SeqPair(seq0='C3A', count0=3.0, seq1='T5A', count1=1.0, period=6)
    ]

    assert(train_pairings == train_pairings_expected)

    shutil.rmtree(testdir)

    train_data = sequence_feature_data_utils.PairwiseBucketizedDataset(
        train_pairings, ref, randint_functor=lambda *args, **kwargs: 1,
    )

    train_sequences = [["[CLS]"] + list(x) + ["[SEP]"] for x in ["XAAGT", "XBCGT", "XACGA"]]

    tokenizer = data_processing.Tokenizer()
    train_tokenized = [[tokenizer.mapper[i] for i in x] for x in train_sequences]

    def frac_eq(x, y, eps=1e-4):
        return x - eps <= y < x + eps

    def tensor_eq_list(t, l):
        return t.tolist() == l

    assert(len(train_data) == 2), f"{len(val_data)} != 1"

    res0 = train_data[0]
    assert(tensor_eq_list(res0[0][0], train_tokenized[0])), f"{res0[0][0]} != {train_tokenized[0]}"
    assert(tensor_eq_list(res0[0][1], train_tokenized[1])), f"{res0[0][1]} != {train_tokenized[1]}"
    assert(frac_eq(res0[1], 1/4))

    res1 = train_data[1]
    assert(tensor_eq_list(res1[0][0], train_tokenized[0])), f"{res1[0][0]} != {train_tokenized[0]}"
    assert(tensor_eq_list(res1[0][1], train_tokenized[2])), f"{res1[0][1]} != {train_tokenized[2]}"
    assert(frac_eq(res1[1], 1/4))

    print("Test test_pairwise_bucketized_dataset passed")


def test_pairwise_bucketized_dataset_validate_pairings():
    ref = "XACGT"

    testdir = "/tmp/test_pairwise_bucketized_dataset_validate_pairings"
    orig_min_lead_time = create_occurrence_buckets._MIN_LEAD_TIME
    create_occurrence_buckets._MIN_LEAD_TIME = 0

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    """
    A2B   -- Train, 1500
    C3A   -- Val, 1000
    G4del -- Test, 500

    T5A -- Train, 250
    X1Z -- Val, 125
    X1F -- Test, 63
    """
    # A2B will fail
    A2BEurope = 1400
    A2BnEurope = 100

    C3AEurope = 500
    C3AnEurope = 500

    G4delEurope = 250
    G4delnEurope = 250

    T5AEurope = 125
    T5AnEurope = 125

    X1ZEurope = 63
    X1ZnEurope = 62

    X1FEurope = 31
    X1FnEurope = 32

    total = 1500 + 1000 + 500 + 250 + 125 + 63

    df = pandas.DataFrame.from_dict(
        {
            "AA Substitutions": \
                ["Spike_A2B"] * A2BEurope + \
                ["Spike_A2B"] * A2BnEurope +\
                ["Spike_C3A"] * C3AEurope + \
                ["Spike_C3A"] * C3AnEurope + \
                ["Spike_G4del"] * G4delEurope + \
                ["Spike_G4del"] * G4delnEurope + \
                ["Spike_T5A"] * T5AEurope + \
                ["Spike_T5A"] * T5AnEurope + \
                ["Spike_X1Z"] * X1ZEurope + \
                ["Spike_X1Z"] * X1ZnEurope + \
                ["Spike_X1F"] * X1FEurope + \
                ["Spike_X1F"] * X1FnEurope,
            "Submission date": ["2021-01-01"] * total,
            "Pango lineage": ["A"] * total,
            "Location": \
                ["Europe"] * A2BEurope + \
                ["Asia"] * A2BnEurope +\
                ["Europe"] * C3AEurope + \
                ["Asia"] * C3AnEurope + \
                ["Europe"] * G4delEurope + \
                ["Asia"] * G4delnEurope + \
                ["Europe"] * T5AEurope + \
                ["Asia"] * T5AnEurope + \
                ["Europe"] * X1ZEurope + \
                ["Asia"] * X1ZnEurope + \
                ["Europe"] * X1FEurope + \
                ["Asia"] * X1FnEurope,
            "Host": ["human"] * total,
            "reconstruction_success_Spike": [True] * total,
            "Accession ID": list(range(total)),
        }
    )
    tsv = os.path.join(testdir, "test.tsv")
    df.to_csv(tsv, sep="\t")

    def nothing_function(*args, **kwargs):
        pass

    results = create_occurrence_buckets.get_training_pairs_from_tsv(
        tsv=tsv,
        availability_last_date="2021-01-01",
        primary_locations=["Europe"],
        control_locations=["Europe"],
        exclude_control=True,
        train_seqs_per_bucket=1,
        val_seqs_per_bucket=1,
        test_seqs_per_bucket=1,
        randshuffle_functor=nothing_function,
        num_randomizations=1,
        p_value=0.1,
    )

    train_pairings, val_pairings, test_pairings = results
    
    assert(not train_pairings)
    assert(len(val_pairings) == len(test_pairings) == 1)
    assert(all([
        val_pairings[0].seq0 == "C3A",
        val_pairings[0].seq1 == "X1Z",
        val_pairings[0].count0 == 1000,
        val_pairings[0].count1 == 125,
        val_pairings[0].period == 1
    ]))
    assert(all([
        test_pairings[0].seq0 == "G4del",
        test_pairings[0].seq1 == "X1F",
        test_pairings[0].count0 == 500,
        test_pairings[0].count1 == 63,
        test_pairings[0].period == 1
    ]))

    create_occurrence_buckets._MIN_LEAD_TIME = orig_min_lead_time

    print("Test test_pairwise_bucketized_dataset_validate_pairings passed")


def test_pairwise_bucketized_dataset_validate_pairings_pre_sequences():
    ref = "XACGT"

    testdir = "/tmp/test_pairwise_bucketized_dataset_validate_pairings_pre_sequences"
    orig_min_lead_time = create_occurrence_buckets._MIN_LEAD_TIME
    create_occurrence_buckets._MIN_LEAD_TIME = 0

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    """
    By default (see test_pairwise_bucketized_dataset_validate_pairings):
    A2B   -- Train, 1500
    C3A   -- Val, 1000
    G4del -- Test, 500

    T5A -- Train, 250
    X1Z -- Val, 125
    X1F -- Test, 63

    We will force this to be the following instead:
    C3A, X1Z -> Train
    A2B, T5A -> Val
    G4del, X1F -> Test
    """
    def pre_sequences_write_helper(items: list, fname):
        with open(os.path.join(testdir, fname), "w") as fhandle:
            for i in items:
                fhandle.write(f"{i}\n")
            name = fhandle.name
        return name

    train_pre = pre_sequences_write_helper(["C3A", "X1Z"], "train_sequences.lst")
    val_pre = pre_sequences_write_helper(["A2B", "T5A"], "val_sequences.lst")
    test_pre = pre_sequences_write_helper(["G4del", "X1F"], "test_sequences.lst")

    # A2B will fail
    A2BEurope = 1400
    A2BnEurope = 100

    C3AEurope = 500
    C3AnEurope = 500

    G4delEurope = 250
    G4delnEurope = 250

    T5AEurope = 125
    T5AnEurope = 125

    X1ZEurope = 63
    X1ZnEurope = 62

    X1FEurope = 31
    X1FnEurope = 32

    # Add C3A cases outside the cutoff date
    C3AEurope_invalid = 501
    C3AnEurope_invalid = 499
    G4delEurope_invalid = 252
    G4delnEurope_invalid = 249
    T5AEurope_invalid = 126
    T5AnEurope_invalid = 129
    X1ZEurope_invalid = 64
    X1ZnEurope_invalid = 61
    X1FEurope_invalid = 36
    X1FnEurope_invalid = 31

    total = 1500 + 1000 + 500 + 250 + 125 + 63

    total_ = 501 + 499 + 252 + 249 + 126 + 129 + 64 + 61 + 36 + 31

    df = pandas.DataFrame.from_dict(
        {
            "AA Substitutions": \
                ["Spike_A2B"] * A2BEurope + \
                ["Spike_A2B"] * A2BnEurope +\
                ["Spike_C3A"] * C3AEurope + \
                ["Spike_C3A"] * C3AnEurope + \
                ["Spike_G4del"] * G4delEurope + \
                ["Spike_G4del"] * G4delnEurope + \
                ["Spike_T5A"] * T5AEurope + \
                ["Spike_T5A"] * T5AnEurope + \
                ["Spike_X1Z"] * X1ZEurope + \
                ["Spike_X1Z"] * X1ZnEurope + \
                ["Spike_X1F"] * X1FEurope + \
                ["Spike_X1F"] * X1FnEurope + \
                # Sequences added after cutoff date are not used
                ["Spike_C3A"] * C3AEurope_invalid + \
                ["Spike_C3A"] * C3AnEurope_invalid + \
                ["Spike_G4del"] * G4delEurope_invalid + \
                ["Spike_G4del"] * G4delnEurope_invalid + \
                ["Spike_T5A"] * T5AEurope_invalid + \
                ["Spike_T5A"] * T5AnEurope_invalid + \
                ["Spike_X1Z"] * X1ZEurope_invalid + \
                ["Spike_X1Z"] * X1ZnEurope_invalid + \
                ["Spike_X1F"] * X1FEurope_invalid + \
                ["Spike_X1F"] * X1FnEurope_invalid,
            "Submission date": ["2021-01-01"] * total + ["2021-02-14"] * total_,
            "Pango lineage": ["A"] * (total + total_),
            "Location": \
                ["Europe"] * A2BEurope + \
                ["Asia"] * A2BnEurope +\
                ["Europe"] * C3AEurope + \
                ["Asia"] * C3AnEurope + \
                ["Europe"] * G4delEurope + \
                ["Asia"] * G4delnEurope + \
                ["Europe"] * T5AEurope + \
                ["Asia"] * T5AnEurope + \
                ["Europe"] * X1ZEurope + \
                ["Asia"] * X1ZnEurope + \
                ["Europe"] * X1FEurope + \
                ["Asia"] * X1FnEurope + \
                # Sequences submitted after deadline are not used
                ["Europe"] * C3AEurope_invalid + \
                ["Asia"] * C3AnEurope_invalid + \
                ["Europe"] * G4delEurope_invalid + \
                ["Asia"] * G4delnEurope_invalid + \
                ["Europe"] * T5AEurope_invalid + \
                ["Asia"] * T5AnEurope_invalid + \
                ["Europe"] * X1ZEurope_invalid + \
                ["Asia"] * X1ZnEurope_invalid + \
                ["Europe"] * X1FEurope_invalid + \
                ["Asia"] * X1FnEurope_invalid,
            "Host": ["human"] * (total + total_),
            "reconstruction_success_Spike": [True] * (total + total_),
            "Accession ID": list(range(total + total_)),
        }
    )
    tsv = os.path.join(testdir, "test.tsv")
    df.to_csv(tsv, sep="\t")

    def nothing_function(*args, **kwargs):
        pass

    results = create_occurrence_buckets.get_training_pairs_from_tsv(
        tsv=tsv,
        availability_last_date="2021-01-01",
        primary_locations=["Europe"],
        control_locations=["Europe"],
        exclude_control=True,
        train_seqs_per_bucket=1,
        val_seqs_per_bucket=1,
        test_seqs_per_bucket=1,
        randshuffle_functor=nothing_function,
        num_randomizations=1,
        p_value=0.1,
        train_sequences_pre=train_pre,
        val_sequences_pre=val_pre,
        test_sequences_pre=test_pre,
    )

    train_pairings, val_pairings, test_pairings = results
    
    assert(not val_pairings)
    assert(len(train_pairings) == len(test_pairings) == 1)
    assert(all([
        train_pairings[0].seq0 == "C3A",
        train_pairings[0].seq1 == "X1Z",
        train_pairings[0].count0 == 1000,
        train_pairings[0].count1 == 125,
        train_pairings[0].period == 1
    ]))
    assert(all([
        test_pairings[0].seq0 == "G4del",
        test_pairings[0].seq1 == "X1F",
        test_pairings[0].count0 == 500,
        test_pairings[0].count1 == 63,
        test_pairings[0].period == 1
    ]))

    create_occurrence_buckets._MIN_LEAD_TIME = orig_min_lead_time

    print("Test test_pairwise_bucketized_dataset_validate_pairings_pre_sequences passed")


def test_combine_mutations():
    import copy
    from dataclasses import asdict
    t1 = [
        create_occurrence_buckets.MutationDetails(
            mutation="A2B",
            count=np.array([0, 1, 1, 1, 1]),
            week=0,
            last_week=4,
        )
    ]
    t2 = []

    orig_t1 = copy.deepcopy(t1)
    orig_t2 = copy.deepcopy(t2)

    def dataclass_eq(a: create_occurrence_buckets.MutationDetails, b: create_occurrence_buckets.MutationDetails) -> bool:
        da = asdict(a)
        db = asdict(b)
        flag = da["mutation"] == db["mutation"]
        flag = flag and np.all(da["count"] == db["count"])
        flag = flag and (da["week"] == db["week"])
        flag = flag and (da["last_week"] == db["last_week"])
        return flag

    def dataclass_list_eq(a: list, b: list) -> bool:
        return (len(a) == len(b)) and all(dataclass_eq(i, j) for i, j in zip(a, b))

    res = create_occurrence_buckets.combine_multiple_data(t1, t2, combine_type="intersection")
    assert(not res)
    assert(dataclass_list_eq(t1, orig_t1))
    assert(dataclass_list_eq(t2, orig_t2))

    res = create_occurrence_buckets.combine_multiple_data(t1, t2, combine_type="union")
    exp_t2 = [
        create_occurrence_buckets.MutationDetails(
            mutation="A2B",
            count=np.array([0, 0, 0, 0, 0]),
            week=0,
            last_week=4,
        )
    ]
    assert(dataclass_list_eq(res, orig_t1))
    assert(dataclass_list_eq(t1, orig_t1))
    assert(dataclass_list_eq(t2, exp_t2)), f"{t2} != {exp_t2}"

    print("Test test_combine_mutations passed")


def test_pairwise_bucketized_dataset_validate_pairings_intersection_vs_union():
    ref = "XACGT"

    testdir = "/tmp/test_pairwise_bucketized_dataset_validate_pairings_intersection_vs_union"
    orig_min_lead_time = create_occurrence_buckets._MIN_LEAD_TIME
    create_occurrence_buckets._MIN_LEAD_TIME = 0

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    """
    A2B   -- Train, 1500
    C3A   -- Val, 1000
    G4del -- Test, 500

    T5A -- Train, 250
    X1Z -- Val, 125
    X1F -- Test, 63
    """
    # A2B will fail
    A2BEurope = 1400
    A2BnEurope = 100

    C3AEurope = 500
    C3AnEurope = 500

    G4delEurope = 250
    G4delnEurope = 250

    T5AEurope = 125
    T5AnEurope = 125

    X1ZEurope = 63
    X1ZnEurope = 62

    X1FEurope = 2
    X1FnEurope = 0

    total = 1500 + 1000 + 500 + 250 + 125 + (X1FEurope + X1FnEurope)

    df = pandas.DataFrame.from_dict(
        {
            "AA Substitutions": \
                ["Spike_A2B"] * A2BEurope + \
                ["Spike_A2B"] * A2BnEurope +\
                ["Spike_C3A"] * C3AEurope + \
                ["Spike_C3A"] * C3AnEurope + \
                ["Spike_G4del"] * G4delEurope + \
                ["Spike_G4del"] * G4delnEurope + \
                ["Spike_T5A"] * T5AEurope + \
                ["Spike_T5A"] * T5AnEurope + \
                ["Spike_X1Z"] * X1ZEurope + \
                ["Spike_X1Z"] * X1ZnEurope + \
                ["Spike_X1F"] * X1FEurope + \
                ["Spike_X1F"] * X1FnEurope,
            "Submission date": ["2021-01-01"] * total,
            "Pango lineage": ["A"] * total,
            "Location": \
                ["Europe"] * A2BEurope + \
                ["Asia"] * A2BnEurope +\
                ["Europe"] * C3AEurope + \
                ["Asia"] * C3AnEurope + \
                ["Europe"] * G4delEurope + \
                ["Asia"] * G4delnEurope + \
                ["Europe"] * T5AEurope + \
                ["Asia"] * T5AnEurope + \
                ["Europe"] * X1ZEurope + \
                ["Asia"] * X1ZnEurope + \
                ["Europe"] * X1FEurope + \
                ["Asia"] * X1FnEurope,
            "Host": ["human"] * total,
            "reconstruction_success_Spike": [True] * total,
            "Accession ID": list(range(total)),
        }
    )
    tsv = os.path.join(testdir, "test.tsv")
    df.to_csv(tsv, sep="\t")

    def nothing_function(*args, **kwargs):
        pass

    results_intersection = create_occurrence_buckets.get_training_pairs_from_tsv(
        tsv=tsv,
        availability_last_date="2021-01-01",
        primary_locations=["Europe"],
        control_locations=["Europe"],
        exclude_control=True,
        train_seqs_per_bucket=1,
        val_seqs_per_bucket=1,
        test_seqs_per_bucket=1,
        randshuffle_functor=nothing_function,
        num_randomizations=1,
        p_value=0.1,
    )

    train_pairings, val_pairings, test_pairings = results_intersection
    
    assert(not train_pairings)
    assert(len(val_pairings) == 1)
    assert(all([
        val_pairings[0].seq0 == "C3A",
        val_pairings[0].seq1 == "X1Z",
        val_pairings[0].count0 == 1000,
        val_pairings[0].count1 == 125,
        val_pairings[0].period == 1
    ]))
    assert(not test_pairings)

    results_union = create_occurrence_buckets.get_training_pairs_from_tsv(
        tsv=tsv,
        availability_last_date="2021-01-01",
        primary_locations=["Europe"],
        control_locations=["Europe"],
        exclude_control=True,
        train_seqs_per_bucket=1,
        val_seqs_per_bucket=1,
        test_seqs_per_bucket=1,
        randshuffle_functor=nothing_function,
        num_randomizations=1,
        p_value=0.1,
        combine_type="union",
    )
    train_pairings, val_pairings, test_pairings = results_union

    assert(not train_pairings)
    assert(len(val_pairings) == len(test_pairings) == 1)
    assert(all([
        val_pairings[0].seq0 == "C3A",
        val_pairings[0].seq1 == "X1Z",
        val_pairings[0].count0 == 1000,
        val_pairings[0].count1 == 125,
        val_pairings[0].period == 1
    ]))
    assert(all([
        test_pairings[0].seq0 == "G4del",
        test_pairings[0].seq1 == "X1F",
        test_pairings[0].count0 == 500,
        test_pairings[0].count1 == 2,
        test_pairings[0].period == 1
    ]))

    create_occurrence_buckets._MIN_LEAD_TIME = orig_min_lead_time

    print("Test test_pairwise_bucketized_dataset_validate_pairings_intersection_vs_union passed")


def test_hi_lo_pairings():
    testdir = "/tmp/test_hi_lo_pairings"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    A2BEurope = 1400
    A2BnEurope = 100

    C3AEurope = 500
    C3AnEurope = 500

    G4delEurope = 250
    G4delnEurope = 250

    T5AEurope = 125
    T5AnEurope = 125

    X1ZEurope = 1
    X1ZnEurope = 2

    X1FEurope = 3
    X1FnEurope = 1

    X1BEurope = 1
    X1BnEurope = 2

    X1CEurope = 2
    X1CnEurope = 2

    X1DEurope = 1
    X1DnEurope = 1

    total = 1500 + 1000 + 500 + 250 + 3 + 4 + 3 + 4 + 2

    df = pandas.DataFrame.from_dict(
        {
            "AA Substitutions": \
                ["Spike_A2B"] * A2BEurope + \
                ["Spike_A2B"] * A2BnEurope +\
                ["Spike_C3A"] * C3AEurope + \
                ["Spike_C3A"] * C3AnEurope + \
                ["Spike_G4del"] * G4delEurope + \
                ["Spike_G4del"] * G4delnEurope + \
                ["Spike_T5A"] * T5AEurope + \
                ["Spike_T5A"] * T5AnEurope + \
                ["Spike_X1Z"] * X1ZEurope + \
                ["Spike_X1Z"] * X1ZnEurope + \
                ["Spike_X1F"] * X1FEurope + \
                ["Spike_X1F"] * X1FnEurope + \
                ["Spike_X1B"] * (X1BEurope + X1BnEurope) + \
                ["Spike_X1C"] * (X1CEurope + X1CnEurope) + \
                ["Spike_X1D"] * (X1DEurope + X1DnEurope),
            "Submission date": ["2021-01-01"] * (A2BEurope + A2BnEurope) + \
                ["2021-01-08"] * (C3AEurope + C3AnEurope) + \
                ["2021-01-15"] * (G4delEurope + G4delnEurope) + \
                ["2021-01-22"] * (T5AEurope + T5AnEurope) + \
                ["2021-01-29"] * (X1ZEurope + X1ZnEurope) + \
                ["2021-02-15"] * (X1FEurope + X1FnEurope) + \
                ["2021-02-22"] * (X1BEurope + X1BnEurope) + \
                ["2021-03-01"] * (X1CEurope + X1CnEurope) + \
                ["2021-03-15"] * (X1DEurope + X1DnEurope),
            "Pango lineage": ["A"] * total,
            "Location": \
                ["Europe"] * A2BEurope + \
                ["Asia"] * A2BnEurope +\
                ["Europe"] * C3AEurope + \
                ["Asia"] * C3AnEurope + \
                ["Europe"] * G4delEurope + \
                ["Asia"] * G4delnEurope + \
                ["Europe"] * T5AEurope + \
                ["Asia"] * T5AnEurope + \
                ["Europe"] * X1ZEurope + \
                ["Asia"] * X1ZnEurope + \
                ["Europe"] * X1FEurope + \
                ["Asia"] * X1FnEurope + \
                ["Europe"] * X1BEurope + \
                ["Asia"] * X1BnEurope + \
                ["Europe"] * X1CEurope + \
                ["Asia"] * X1CnEurope + \
                ["Europe"] * X1DEurope + \
                ["Asia"] * X1DnEurope,
            "Host": ["human"] * total,
            "reconstruction_success_Spike": [True] * total,
            "Accession ID": list(range(total)),
        }
    )
    tsv = os.path.join(testdir, "test.tsv")
    df.to_csv(tsv, sep="\t")

    df_with_weeks, min_date = create_occurrence_buckets.prepare_tsv(
        tsv,
        availability_last_date="2021-04-28",
    )

    freq_seqs, rare_seqs = create_occurrence_buckets.get_hi_lo_pairings(
        df_with_weeks, n_trailing_weeks=1, min_count=100, max_total=5
    )

    assert("X1D" not in [x[0] for x in rare_seqs])
    assert(max(x[1] for x in rare_seqs) <= 5)

    train_pairings = [
        create_occurrence_buckets.SeqPair(
            seq0="A2B", count0=1500, seq1="X1Z", count1=1, period=100),
    ]

    def nothing_fn(*args, **kwargs):
        pass

    train_pairings, val_pairings, test_pairings = create_occurrence_buckets.add_hi_lo_pairings(
        train_pairings, [], [],
        freq_seqs, rare_seqs,
        train_seqs_per_bucket=1, val_seqs_per_bucket=1, test_seqs_per_bucket=1,
        randshuffle_functor=nothing_fn,
    )

    # train: [A2B, remaining_freq[0]] x [X1Z, remaining_rare[0]]
    # val: [remaining_freq[1]] x [remaining_rare[1]]
    # test: [remaining_freq[2]] x [remaining_rare[2]]
    remaining_freq = [x for x in freq_seqs if x[0] != "A2B"]
    remaining_rare = [x for x in rare_seqs if x[1] != "X1Z"]

    train_exp = []

    for i in itertools.product([("A2B", 1500), remaining_freq[0]], [("X1Z", 3), remaining_rare[0]]):
        train_exp.append(create_occurrence_buckets.SeqPair(
            seq0=i[0][0],
            count0=i[0][1],
            seq1=i[1][0],
            count1=i[1][1],
            period=None
        ))

    val_exp = [
        create_occurrence_buckets.SeqPair(
            seq0=remaining_freq[1][0],
            count0=remaining_freq[1][1],
            seq1=remaining_rare[1][0],
            count1=remaining_rare[1][1],
            period=None,
        )
    ]

    test_exp = [
        create_occurrence_buckets.SeqPair(
            seq0=remaining_freq[2][0],
            count0=remaining_freq[2][1],
            seq1=remaining_rare[2][0],
            count1=remaining_rare[2][1],
            period=None,
        )
    ]

    assert(train_exp == train_pairings)
    assert(val_exp == val_pairings)
    assert(test_exp == test_pairings)

    print("Test test_hi_lo_pairings passed")


def test_fake_sequence_addition():
    class RandSampleTest:
        def __init__(self):
            self.invocation_counter = 0

        def __call__(self, pop, *args, **kwargs):
            if self.invocation_counter % 2 == 0:
                r = ["F6Z", "C3D", "A2B"]
            else:
                r = ["D4E", "E5G", "C3D"]

            self.invocation_counter += 1

            return r

    df = pandas.DataFrame.from_dict(
        {
            "SpikeMutations": [
                "A2B,F6Z", "C3D", "D4E", "E5G"
            ]
        }
    )
    train_pairings = [
        create_occurrence_buckets.SeqPair(
            seq0="F6B,A1024G",
            count0=100,
            seq1="E5A",
            count1=1,
            period=1,
        )
    ]
    a, b = create_occurrence_buckets.add_fake_sequences_top(
        df,
        train_pairings,
        n_random_mutations_for_negatives=3,
        n_negative_trials_per_sequence=2,
        randsample_functor=RandSampleTest(),
    )
    exp_all_sequences = set({"A2B,F6Z", "C3D", "D4E", "E5G"})
    exp_all_mutations = set({"A2B", "F6Z", "C3D", "D4E", "E5G"})
    assert(exp_all_sequences == a)
    assert(exp_all_mutations == set(b))
    exp_pairings = train_pairings + [
        create_occurrence_buckets.SeqPair(
            seq0="F6B,A1024G",
            count0=1,
            seq1="C3D,D4E,E5G,F6B,A1024G",
            count1=0,
            period=None,
        ),
        create_occurrence_buckets.SeqPair(
            seq0="E5A",
            count0=1,
            seq1="A2B,C3D,E5A,F6Z",
            count1=0,
            period=None,
        )
    ]
    assert(set(exp_pairings) == set(train_pairings))
    print("Test test_fake_sequence_addition passed")


def test_tokenize_helper():
    import utils
    mapper = data_processing.Tokenizer().mapper
    special_tokens = utils._DEFAULT_SPECIAL_TOKENS
    assert_test_passed = False

    try:
        res0 = sequence_feature_data_utils.tokenize_helper(
            seq="A2B,C3A",
            ref="CACG",
            special_tokens=special_tokens,
            mapper=mapper,
        )
    except AssertionError:
        assert_test_passed = True

    assert(assert_test_passed), "No assertion was thrown"

    res1 = sequence_feature_data_utils.tokenize_helper(
        seq="A2B,C3A",
        ref="CACG*",
        special_tokens=special_tokens,
        mapper=mapper,
    )
    expected = [mapper["[CLS]"], mapper["C"], mapper["B"], mapper["A"], mapper["G"], mapper["[SEP]"]]
    assert(res1 == expected)
    print("Test test_tokenize_helper passed")


def test_train_val_split_precalculated():
    dates_and_counts = [
        create_occurrence_buckets.MutationDetails(
            mutation="A2B",
            count=np.array([1, 1, 1, 2]),
            week=6,
            last_week=9,
        ),
        create_occurrence_buckets.MutationDetails(
            mutation="C3G",
            count=np.array([2, 1, 1, 2]),
            week=7,
            last_week=10,
        ),
        create_occurrence_buckets.MutationDetails(
            mutation="T4G",
            count=np.array([3, 2, 1, 2]),
            week=8,
            last_week=11,
        )
    ]

    train_sequences = ["A2B"]
    val_sequences = ["C3G"]
    test_sequences = ["T4G"]

    results = create_occurrence_buckets.train_val_split_precalculated(
        dates_and_counts, train_sequences, val_sequences, test_sequences)

    assert(results[0] == dates_and_counts[0:1])
    assert(results[1] == dates_and_counts[1:2])
    assert(results[2] == dates_and_counts[2:3])

    print("Test test_train_val_split_precalculated passed")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_create_embeddings_data(testdir="/tmp/test_create_embeddings_data")
    shutil.rmtree("/tmp/test_create_embeddings_data")
    test_pairwise_dataset()
    test_pairwise_dataset_embeddings()
    test_pairwise_bucketized_dataset()
    test_pairwise_bucketized_dataset_validate_pairings()
    test_hi_lo_pairings()
    test_fake_sequence_addition()
    test_tokenize_helper()
    test_train_val_split_precalculated()
    test_combine_mutations()
    test_pairwise_bucketized_dataset_validate_pairings_intersection_vs_union()
    test_pairwise_bucketized_dataset_validate_pairings_pre_sequences()
    
