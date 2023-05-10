# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import uniprot_training_data_dump


def test_uniprot_training_data_dump():
    import data_processing
    import argparse
    import shutil
    import os
    import json

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

    testdir = "/tmp/test_uniprot_training_data_dump"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    args = argparse.Namespace(
        train_val_test_split=",".join([str(x) for x in (0.98, 0.01, 0.01)]),
        max_sequence_length=12,
        max_masked_segment=3,
        include_extremities=False,
        datadir=testdir,
        fasta="test_fasta.fasta",
        max_items_to_process=None,
        num_workers=2,
    )

    uniprot_training_data_dump._TEST_RANDSAMPLE_FUNCTOR = RandSampleFunctor()
    uniprot_training_data_dump._TEST_RANDINT_FUNCTOR = RandIntFunctor()
    uniprot_training_data_dump._TEST_RANDCHOICES_FUNCTOR = RandChoicesFunctor()

    uniprot_training_data_dump.main(args)

    # Train checks (ACGXXFNMBMNUACHUACPQRST, GCRXRRNMA)
    full_train_sequence = "ACGXXFNMBMNUACHUACPQRST"
    train_sequences = [full_train_sequence[: 12], full_train_sequence[12: ], "GCRXRRNMA"]
    tokenizer_results = [tokenizer.tokenize(x) for x in train_sequences]
    train_data = data_processing.DiskStorage.load(os.path.join(testdir, "train"), mode="r")

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
    val_data = data_processing.DiskStorage.load(os.path.join(testdir, "val"), mode="r")
    res0 = val_data[0]
    assert(compare(res0.itokens, tokenizer_results[0][0])), (res0, tokenizer_results[0][0])
    assert(compare(res0.otokens, tokenizer_results[0][1])), (res0, tokenizer_results[0][1])
    assert(json.loads(res0.metadata) == {"uniref_id": "ABCD3", "fragment": "0"})

    # Test data checks (GCRXMFNMA)
    test_sequences = ["GCRXMFNMA"]
    tokenizer_results = [tokenizer.tokenize(x) for x in test_sequences]
    test_data = data_processing.DiskStorage.load(os.path.join(testdir, "test"), mode="r")
    res0 = test_data[0]
    assert(compare(res0.itokens, tokenizer_results[0][0])), (res0, tokenizer_results[0][0])
    assert(compare(res0.otokens, tokenizer_results[0][1])), (res0, tokenizer_results[0][1])
    assert(json.loads(res0.metadata) == {"uniref_id": "ABCD1", "fragment": "0"})

    print("Test test_uniprot_training_data_dump passed")


if __name__ == "__main__":
    test_uniprot_training_data_dump()