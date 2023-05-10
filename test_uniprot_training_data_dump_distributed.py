# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import uniprot_training_data_dump_distributed
import pickle


def test_uniprot_training_data_dump_distributed():
    import data_processing
    import argparse
    import shutil
    import os
    import json
    import subprocess

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

    tokenizer = data_processing.Tokenizer(
        max_sequence_length=1024,
        max_masked_segment=1024,
        include_extremities=False,
        randsample_functor=RandSampleFunctor(),
        randint_functor=RandIntFunctor(),
    )

    testdir = "/tmp/test_uniprot_training_data_dump_distributed"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    # uniprot_training_data_dump.main(args)
    thisdir = os.path.split(__file__)[0]
    subprocess.run(
        ["torchrun", "--nproc_per_node", "2", "--nnodes", "1",
        os.path.join(thisdir, "uniprot_training_data_dump_distributed.py"),
        "--max_sequence_length", "12",
        "--max_masked_segment", "3",
        "--datadir", testdir,
        "--fasta", os.path.join(thisdir, "test_fasta.fasta"),
        "--test",
        ], check=True
    )

    # Train checks (ACGXXFNMBMNUACHUACPQRST, GCRXRRNMA)
    full_train_sequence = "ACGXXFNMBMNUACHUACPQRST"
    train_sequences = [full_train_sequence[: 12], full_train_sequence[12: ], "GCRXRRNMA"]
    tokenizer_results = [tokenizer.tokenize(x) for x in train_sequences]
    train_data = data_processing.CompoundDiskStorageReader(
        datadir=testdir, data_type="train")
    assert(len(train_data) == 3)

    def compare(actual, expected):
        actual_list = actual["input_ids"][actual["attention_mask"] == 1].tolist()
        return actual_list == expected

    def compare_helper(res):
        metadata = json.loads(res.metadata)
        if metadata["uniref_id"] == "ABCD0":
            fragment = metadata["fragment"]
            if fragment == "0":
                assert(compare(res.itokens, tokenizer_results[0][0])), (res, tokenizer_results[0][0])
                assert(compare(res.otokens, tokenizer_results[0][1])), (res, tokenizer_results[0][1])
            elif fragment == "1":
                assert(compare(res.itokens, tokenizer_results[1][0])), (res, tokenizer_results[1][0])
                assert(compare(res.otokens, tokenizer_results[1][1])), (res, tokenizer_results[1][1])
            else:
                raise ValueError(f"Unknown fragment {fragment}")
        elif metadata["uniref_id"] == "ABCD2a":
            assert(metadata["fragment"] == "0"), (res)
            assert(compare(res.itokens, tokenizer_results[2][0])), (res, tokenizer_results[2][0])
            assert(compare(res.otokens, tokenizer_results[2][1])), (res, tokenizer_results[2][1])
        else:
            raise ValueError(f"Unknown uniref_id")

    compare_helper(train_data[0])
    compare_helper(train_data[1])
    compare_helper(train_data[2])

    # Validation checks (ACBDEGFHIJKL)
    val_sequences = ["ACBDEGFHIJKL"]
    tokenizer_results = [tokenizer.tokenize(x) for x in val_sequences]
    val_data = data_processing.CompoundDiskStorageReader(
        testdir, data_type="val")
    res0 = val_data[0]
    assert(compare(res0.itokens, tokenizer_results[0][0])), (res0, tokenizer_results[0][0])
    assert(compare(res0.otokens, tokenizer_results[0][1])), (res0, tokenizer_results[0][1])
    assert(json.loads(res0.metadata) == {"uniref_id": "ABCD3", "fragment": "0"})

    # Test data checks (GCRXMFNMA)
    test_sequences = ["GCRXMFNMA"]
    tokenizer_results = [tokenizer.tokenize(x) for x in test_sequences]
    test_data = data_processing.CompoundDiskStorageReader(
        testdir, data_type="test")
    res0 = test_data[0]
    assert(compare(res0.itokens, tokenizer_results[0][0])), (res0, tokenizer_results[0][0])
    assert(compare(res0.otokens, tokenizer_results[0][1])), (res0, tokenizer_results[0][1])
    assert(json.loads(res0.metadata) == {"uniref_id": "ABCD1", "fragment": "0"})

    print("Test test_uniprot_training_data_dump_distributed passed")


def test_uniprot_training_data_dump_distributed_uniref100():
    import data_processing
    import argparse
    import shutil
    import os
    import json
    import subprocess

    """
    "ax" maps to 2
    "av" maps to 1
    "az" maps to 0
    "ab" maps to 0
    """

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

    tokenizer = data_processing.Tokenizer(
        max_sequence_length=1024,
        max_masked_segment=1024,
        include_extremities=False,
        randsample_functor=RandSampleFunctor(),
        randint_functor=RandIntFunctor(),
    )

    testdir = "/tmp/test_uniprot_training_data_dump_distributed_uniref100"
    clusterdir = "/tmp/test_uniprot_training_data_dump_distributed_uniref100_clusters"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    if os.path.exists(clusterdir):
        shutil.rmtree(clusterdir)

    os.makedirs(clusterdir)
    with open(os.path.join(clusterdir, "cluster_0.uniref100.pkl"), "wb") as fhandle:
        # ABCD0 -> val
        # ABCD1 -> train
        pickle.dump({"ABCD0": "av", "ABCD1": "az"}, fhandle)

    with open(os.path.join(clusterdir, "cluster_1.uniref100.pkl"), "wb") as fhandle:
        # ABCD2a -> test
        pickle.dump({"ABCD2a": "ax"}, fhandle)

    with open(os.path.join(clusterdir, "cluster_2.uniref100.pkl"), "wb") as fhandle:
        # ABCD3 -> train
        pickle.dump({"ABCD3": "ab"}, fhandle)

    thisdir = os.path.split(__file__)[0]
    subprocess.run(
        ["torchrun", "--nproc_per_node", "3", "--nnodes", "1",
        os.path.join(thisdir, "uniprot_training_data_dump_distributed.py"),
        "--max_sequence_length", "12",
        "--max_masked_segment", "3",
        "--datadir", testdir,
        "--fasta", os.path.join(thisdir, "test_fasta.fasta"),
        "--test",
        "--uniref100_cluster_data", os.path.join(clusterdir, "cluster"),
        "--num_sampler_buckets", str(3),
        "--train_val_test_split", "0.33,0.33,0.33",
        ], check=True
    )

    # Train checks
    train_sequences = ["GCRXMFNMA", "ACBDEGFHIJKL"]
    tokenizer_results = [tokenizer.tokenize(x) for x in train_sequences]
    train_data = data_processing.CompoundDiskStorageReader(
        datadir=testdir, data_type="train")
    assert(len(train_data) == 2)

    def compare(actual, expected):
        actual_list = actual["input_ids"][actual["attention_mask"] == 1].tolist()
        return actual_list == expected

    def compare_helper(res):
        metadata = json.loads(res.metadata)
        if metadata["uniref_id"] == "ABCD1":
            fragment = metadata["fragment"]
            assert(metadata["fragment"] == "0"), (res)
            assert(compare(res.itokens, tokenizer_results[0][0])), (res, tokenizer_results[0][0])
            assert(compare(res.otokens, tokenizer_results[0][1])), (res, tokenizer_results[0][1])
        elif metadata["uniref_id"] == "ABCD3":
            assert(metadata["fragment"] == "0"), (res)
            assert(compare(res.itokens, tokenizer_results[1][0])), (res, tokenizer_results[1][0])
            assert(compare(res.otokens, tokenizer_results[1][1])), (res, tokenizer_results[1][1])
        else:
            raise ValueError(f"Unknown uniref_id")

    assert(len(train_data) == 2)
    compare_helper(train_data[0])
    compare_helper(train_data[1])

    # Validation checks
    full_validation_sequence = "ACGXXFNMBMNUACHUACPQRST"
    val_sequences = [full_validation_sequence[:12], full_validation_sequence[12:]]
    tokenizer_results = [tokenizer.tokenize(x) for x in val_sequences]
    val_data = data_processing.CompoundDiskStorageReader(testdir, data_type="val")
    assert(len(val_data) == 2)
    for i in range(2):
        res = val_data[i]
        metadata = json.loads(res.metadata)
        if metadata["fragment"] == "0":
            assert(compare(res.itokens, tokenizer_results[0][0])), (res, tokenizer_results[0][0])
            assert(compare(res.otokens, tokenizer_results[0][1])), (res, tokenizer_results[0][1])
            assert(metadata == {"uniref_id": "ABCD0", "fragment": "0"}), metadata
        elif metadata["fragment"] == "1":
            assert(compare(res.itokens, tokenizer_results[1][0])), (res, tokenizer_results[1][0])
            assert(compare(res.otokens, tokenizer_results[1][1])), (res, tokenizer_results[1][1])
            assert(metadata == {"uniref_id": "ABCD0", "fragment": "1"}), metadata
        else:
            raise ValueError("Unknown fragment")

    # Test data checks
    test_sequences = ["GCRXRRNMA"]
    tokenizer_results = [tokenizer.tokenize(x) for x in test_sequences]
    test_data = data_processing.CompoundDiskStorageReader(
        testdir, data_type="test")
    res0 = test_data[0]
    assert(compare(res0.itokens, tokenizer_results[0][0])), (res0, tokenizer_results[0][0])
    assert(compare(res0.otokens, tokenizer_results[0][1])), (res0, tokenizer_results[0][1])
    assert(json.loads(res0.metadata) == {"uniref_id": "ABCD2a", "fragment": "0"})

    print("Test test_uniprot_training_data_dump_distributed_uniref100 passed")


def test_uniprot_training_data_dump_distributed_uniref50_simple_splitter():
    import data_processing
    import argparse
    import shutil
    import os
    import json
    import subprocess

    """
    "ax" maps to 2
    "av" maps to 1
    "az" maps to 0
    "ab" maps to 0
    """

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

    tokenizer = data_processing.Tokenizer(
        max_sequence_length=1024,
        max_masked_segment=1024,
        include_extremities=False,
        randsample_functor=RandSampleFunctor(),
        randint_functor=RandIntFunctor(),
    )

    testdir = "/tmp/test_uniprot_training_data_dump_distributed_uniref50_predefined"
    clusterdir = "/tmp/test_uniprot_training_data_dump_distributed_uniref50_predefined_clusters"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    if os.path.exists(clusterdir):
        shutil.rmtree(clusterdir)

    os.makedirs(clusterdir)
    with open(os.path.join(clusterdir, "cluster-0.pkl"), "wb") as fhandle:
        # ABCD0 -> val
        # ABCD1 -> train
        # pickle.dump({"ABCD0": "av", "ABCD1": "az"}, fhandle)
        pickle.dump({"ABCD0": "val", "ABCD1": "train"}, fhandle)

    with open(os.path.join(clusterdir, "cluster-1.pkl"), "wb") as fhandle:
        # ABCD2a -> test
        # pickle.dump({"ABCD2a": "ax"}, fhandle)
        pickle.dump({"ABCD2a": "test"}, fhandle)

    with open(os.path.join(clusterdir, "cluster-2.pkl"), "wb") as fhandle:
        # ABCD3 -> train
        # pickle.dump({"ABCD3": "ab"}, fhandle)
        pickle.dump({"ABCD3": "train"}, fhandle)

    thisdir = os.path.split(__file__)[0]
    subprocess.run(
        ["torchrun", "--nproc_per_node", "3", "--nnodes", "1",
        os.path.join(thisdir, "uniprot_training_data_dump_distributed.py"),
        "--max_sequence_length", "12",
        "--max_masked_segment", "3",
        "--datadir", testdir,
        "--fasta", os.path.join(thisdir, "test_fasta.fasta"),
        "--test",
        "--uniref50_cluster_data", os.path.join(clusterdir, "cluster"),
        "--num_sampler_buckets", str(3),
        "--train_val_test_split", "0.33,0.33,0.33",
        ], check=True
    )

    # Train checks
    train_sequences = ["GCRXMFNMA", "ACBDEGFHIJKL"]
    tokenizer_results = [tokenizer.tokenize(x) for x in train_sequences]
    train_data = data_processing.CompoundDiskStorageReader(
        datadir=testdir, data_type="train")
    assert(len(train_data) == 2)

    def compare(actual, expected):
        actual_list = actual["input_ids"][actual["attention_mask"] == 1].tolist()
        return actual_list == expected

    def compare_helper(res):
        metadata = json.loads(res.metadata)
        if metadata["uniref_id"] == "ABCD1":
            fragment = metadata["fragment"]
            assert(metadata["fragment"] == "0"), (res)
            assert(compare(res.itokens, tokenizer_results[0][0])), (res, tokenizer_results[0][0])
            assert(compare(res.otokens, tokenizer_results[0][1])), (res, tokenizer_results[0][1])
        elif metadata["uniref_id"] == "ABCD3":
            assert(metadata["fragment"] == "0"), (res)
            assert(compare(res.itokens, tokenizer_results[1][0])), (res, tokenizer_results[1][0])
            assert(compare(res.otokens, tokenizer_results[1][1])), (res, tokenizer_results[1][1])
        else:
            raise ValueError(f"Unknown uniref_id")

    assert(len(train_data) == 2)
    compare_helper(train_data[0])
    compare_helper(train_data[1])

    # Validation checks
    full_validation_sequence = "ACGXXFNMBMNUACHUACPQRST"
    val_sequences = [full_validation_sequence[:12], full_validation_sequence[12:]]
    tokenizer_results = [tokenizer.tokenize(x) for x in val_sequences]
    val_data = data_processing.CompoundDiskStorageReader(testdir, data_type="val")
    assert(len(val_data) == 2)
    for i in range(2):
        res = val_data[i]
        metadata = json.loads(res.metadata)
        if metadata["fragment"] == "0":
            assert(compare(res.itokens, tokenizer_results[0][0])), (res, tokenizer_results[0][0])
            assert(compare(res.otokens, tokenizer_results[0][1])), (res, tokenizer_results[0][1])
            assert(metadata == {"uniref_id": "ABCD0", "fragment": "0"}), metadata
        elif metadata["fragment"] == "1":
            assert(compare(res.itokens, tokenizer_results[1][0])), (res, tokenizer_results[1][0])
            assert(compare(res.otokens, tokenizer_results[1][1])), (res, tokenizer_results[1][1])
            assert(metadata == {"uniref_id": "ABCD0", "fragment": "1"}), metadata
        else:
            raise ValueError("Unknown fragment")

    # Test data checks
    test_sequences = ["GCRXRRNMA"]
    tokenizer_results = [tokenizer.tokenize(x) for x in test_sequences]
    test_data = data_processing.CompoundDiskStorageReader(
        testdir, data_type="test")
    res0 = test_data[0]
    assert(compare(res0.itokens, tokenizer_results[0][0])), (res0, tokenizer_results[0][0])
    assert(compare(res0.otokens, tokenizer_results[0][1])), (res0, tokenizer_results[0][1])
    assert(json.loads(res0.metadata) == {"uniref_id": "ABCD2a", "fragment": "0"})

    print("Test test_uniprot_training_data_dump_distributed_uniref50_simple_splitter passed")


if __name__ == "__main__":
    test_uniprot_training_data_dump_distributed()
    test_uniprot_training_data_dump_distributed_uniref100()
    test_uniprot_training_data_dump_distributed_uniref50_simple_splitter()