# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import data_processing
from collections.abc import Iterable


def test_create_example_simple():
    """
    Simple test
    """
    sequence = "ABCDEFG"

    def randint_functor(a: int, b: int) -> int:
        assert(a == 1)
        assert(b == 3)
        return 3

    def randsample_functor(a: Iterable, b: int) -> int:
        assert(list(a) == [1, 2, 3]), "%s" % str(a)
        assert(b == 1)
        return [2]

    res = data_processing.create_example(
        sequence=sequence,
        include_extremities=False,
        randint_functor=randint_functor,
        randsample_functor=randsample_functor,
    )

    assert(res[0] == ["[CLS]", "A", "B", "[MASK]", "F", "G", "[SEP]"]), res[0]
    assert(res[1] == ["C", "D", "E", "[MASK]"])

    print("Test test_create_example_simple passed")


def test_create_example_include_extremities():
    """
    Test with include_extremities
    """
    sequence = "ABCAHIJKFGEFG"

    def randint_functor(a: int, b: int) -> int:
        assert(a == 1)
        assert(b == 6)
        return 3

    def randsample_functor(a: Iterable, b: int) -> int:
        assert(list(a) == [2, 3, 4, 5, 6, 7]), "%s" % str(a)
        assert(b == 1)
        return [4]

    res = data_processing.create_example(
        sequence=sequence,
        include_extremities=False,
        randint_functor=randint_functor,
        randsample_functor=randsample_functor,
    )

    assert(res[0] == 
        ["[CLS]", "A", "B", "C", "A", "[MASK]", "K", "F", "G", "E", "F", "G", "[SEP]"]
    ), res[0]
    assert(res[1] == ["H", "I", "J", "[MASK]"]), res[1]

    print("Test test_create_example_include_extremities passed")


def test_disk_storage():
    import os
    import shutil
    import numpy as np
    testdir = os.path.join("/tmp", "test_disk_storage_testdir")
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    data = data_processing.DiskStorage(4, 2, 2, testdir, "w+")
    data.append(
        encoder_data=[1, 2, 3],
        decoder_data=[9, 8],
        metadata="first_item"
    )
    assert(len(data) == 1)
    data.append(
        encoder_data=[11, 12, 13, 14],
        decoder_data=[1],
        metadata="second_item"
    )
    assert(len(data) == 2)
    data.close()
    del data
    data = data_processing.DiskStorage.load(testdir, "r")
    assert(len(data) == 2)
    res0_expected = {
        "itokens": {
            "input_ids": np.array([1, 2, 3, -1], dtype=np.ubyte),
            "attention_mask": np.array([1, 1, 1, 0], dtype=np.ubyte),
        },
        "otokens": {
            "input_ids": np.array([9, 8], dtype=np.ubyte),
            "attention_mask": np.array([1, 1], dtype=np.ubyte)
        },
        "metadata": "first_item"
    }
    res1_expected = {
        "itokens": {
            "input_ids": np.array([11, 12, 13, 14], dtype=np.ubyte),
            "attention_mask": np.array([1, 1, 1, 1], dtype=np.ubyte),
        },
        "otokens": {
            "input_ids": np.array([1, 0], dtype=np.ubyte),
            "attention_mask": np.array([1, 0], dtype=np.ubyte)
        },
        "metadata": "second_item"
    }
    def compare(actual: data_processing.DiskStorageDataItem, expected: dict) -> bool:
        def compare_tokens(exp: dict, act: dict):
            flag = np.logical_and.reduce(exp["attention_mask"] == act["attention_mask"])
            flag = flag and (
                np.logical_and.reduce(
                    np.logical_or(exp["input_ids"] == act["input_ids"], act["attention_mask"] == 0)
                )
            )
            return flag
        flag = compare_tokens(actual.itokens, expected["itokens"])
        flag = flag and compare_tokens(actual.otokens, expected["otokens"])
        flag = flag and (actual.metadata == expected["metadata"])
        return flag

    assert(compare(data[0], res0_expected))
    assert(compare(data[1], res1_expected))
    print("Test test_disk_storage passed")


def test_disk_storage_initialized():
    import os
    import shutil
    import numpy as np
    testdir = os.path.join("/tmp", "test_disk_storage_testdir")
    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    # Initialize and close
    data = data_processing.DiskStorage(4, 2, 2, testdir, "w+")
    data._initialize()
    data.close()
    del data

    # Load initialized data, write and close
    data = data_processing.DiskStorage.load(testdir, mode="r+")
    assert(len(data) == 2)
    data[0] = ([1, 2, 3], [9, 8], "first_item")
    data[1] = ([11, 12, 13, 14], [1], "second_item")
    assert(len(data) == 2)
    data.close()
    del data

    # Load initialized and written data and check values
    data = data_processing.DiskStorage.load(testdir, mode="r")
    assert(len(data) == 2)
    res0_expected = {
        "itokens": {
            "input_ids": np.array([1, 2, 3, -1], dtype=np.ubyte),
            "attention_mask": np.array([1, 1, 1, 0], dtype=np.ubyte),
        },
        "otokens": {
            "input_ids": np.array([9, 8], dtype=np.ubyte),
            "attention_mask": np.array([1, 1], dtype=np.ubyte)
        },
        "metadata": "first_item"
    }
    res1_expected = {
        "itokens": {
            "input_ids": np.array([11, 12, 13, 14], dtype=np.ubyte),
            "attention_mask": np.array([1, 1, 1, 1], dtype=np.ubyte),
        },
        "otokens": {
            "input_ids": np.array([1, 0], dtype=np.ubyte),
            "attention_mask": np.array([1, 0], dtype=np.ubyte)
        },
        "metadata": "second_item"
    }
    def compare(actual: data_processing.DiskStorageDataItem, expected: dict) -> bool:
        def compare_tokens(exp: dict, act: dict):
            flag = np.logical_and.reduce(exp["attention_mask"] == act["attention_mask"])
            flag = flag and (
                np.logical_and.reduce(
                    np.logical_or(exp["input_ids"] == act["input_ids"], act["attention_mask"] == 0)
                )
            )
            return flag
        flag = compare_tokens(actual.itokens, expected["itokens"])
        flag = flag and compare_tokens(actual.otokens, expected["otokens"])
        flag = flag and (actual.metadata == expected["metadata"])
        return flag

    assert(compare(data[0], res0_expected))
    assert(compare(data[1], res1_expected))
    print("Test test_disk_storage_initialized passed")


def test_find_minimal_non_repeating_presuffix():
    string = "AABCDAIGHACGACGAABHGAIGH"
    assert(data_processing.find_minimal_non_repeating_presuffix(
        string, start_length=1, max_length=16, suffix=False
    ) == 4)
    assert(data_processing.find_minimal_non_repeating_presuffix(
        string, start_length=1, max_length=16, suffix=True
    ) == 5)
    print("Test test_find_minimal_non_repeating_presuffix passed")


def test_tokenizer():
    sequence = "ABCDEFG"

    def randint_functor(a: int, b: int) -> int:
        assert(a == 1)
        assert(b == 3)
        return 3

    def randsample_functor(a: Iterable, b: int) -> int:
        assert(list(a) == [1, 2, 3]), "%s" % str(a)
        assert(b == 1)
        return [2]

    tokenizer = data_processing.Tokenizer(
        max_sequence_length=10,
        max_masked_segment=3,
        include_extremities=False,
        randint_functor=randint_functor,
        randsample_functor=randsample_functor,
    )

    res = tokenizer.tokenize(sequence)

    expected = [tokenizer.mapper[i] for i in [
        "[CLS]", "A", "B", "[MASK]", "F", "G", "[SEP]"
    ]]

    assert(res[0] == expected), str(res[0]) + ":::" + str(expected)

    expected = [tokenizer.mapper[i] for i in ["C", "D", "E", "[MASK]"]]

    assert(res[1] == expected), str(res[1]) + ":::" + str(expected)

    print("Test test_tokenizer passed")


if __name__ == "__main__":
    test_create_example_simple()
    test_create_example_include_extremities()
    test_disk_storage()
    test_disk_storage_initialized()
    test_find_minimal_non_repeating_presuffix()
    test_tokenizer()
