# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import sars_cov2_data_processing
import pandas
import os
import shutil
import argparse
import logging
from utils import AMINO_ACIDS
import data_processing

logging.basicConfig(level=logging.DEBUG)

testdir = "/tmp/test_sars_cov2_data_processing_workdir"
ref = "ABCDNDAC"


def create_data():
    data_dict = {
        "Host": ["human", "human"],
        "AA Substitutions": ["Spike_N5Y", "Spike_D6G,Spike_N5Y"],
        "Collection date": ["2019-12-15", "2019-12-16"],
        "Pango lineage": ["A", "B"],
    }
    df = pandas.DataFrame.from_dict(data_dict)

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    tsvname = os.path.join(testdir, "test.tsv")

    df.to_csv(tsvname, sep="\t")

    with open(os.path.join(testdir, "ref_file.txt"), "w") as fhandle:
        ref_file = fhandle.name
        fhandle.write(ref)

    return tsvname, ref_file


def create_data_select_segments():
    data_dict = {
        "Host": ["human", "human"],
        "AA Substitutions": ["Spike_N5del,Spike_D6del,Spike_A7del,Spike_ins1A",  # AA.B.C.D.-.-.-.C"
            "Spike_D6G,Spike_A7G,Spike_N5Y"],
        "Collection date": ["2019-12-15",
            "2019-12-16",
        ],
        "Pango lineage": ["A", "B"],
    }
    df = pandas.DataFrame.from_dict(data_dict)

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    tsvname = os.path.join(testdir, "test.tsv")

    df.to_csv(tsvname, sep="\t")

    with open(os.path.join(testdir, "ref_file.txt"), "w") as fhandle:
        ref_file = fhandle.name
        fhandle.write(ref)

    return tsvname, ref_file


def get_tokenizations_for_sequence(seq: str) -> dict:
    mapper = {
        i: j for j, i in enumerate(AMINO_ACIDS)
    }
    mapper["[CLS]"] = len(mapper)
    mapper["[SEP]"] = len(mapper)
    mapper["[MASK]"] = len(mapper)
    mapper["*"] = mapper["[SEP]"]

    def tokenize(x):
        return [mapper[i] for i in x]

    seq_first_part_encoder = tokenize(["[CLS]", "[MASK]"] + list(seq[5:]) + ["[SEP]"])
    seq_first_part_decoder = tokenize(list(seq[:5]) + ["[MASK]"])
    seq_second_part_encoder = tokenize(["[CLS]"] + list(seq[:5]) + ["[MASK]", "[SEP]"])
    seq_second_part_decoder = tokenize(list(seq[5:]) + ["[MASK]"])

    return [
        (seq_first_part_encoder, seq_first_part_decoder),
        (seq_second_part_encoder, seq_second_part_decoder),
    ]


def check_tensor_against_list(tensor_dict: dict, target: list) -> bool:
    tensor_to_array = tensor_dict["input_ids"].masked_select(tensor_dict["attention_mask"] == 1).long().tolist()
    return tensor_to_array == target


def validate_dset(dset: data_processing.CompoundDiskStorageReader, num_randomizations: int = 1) -> None:
    assert(len(dset) == 2 * num_randomizations)
    for i in range(num_randomizations):
        ret0 = dset[i * 2 + 0]
        ret1 = dset[i * 2 + 1]

        ret0.tensorize()
        ret1.tensorize()

        assert(ret0.metadata == ret1.metadata)

        if ret0.metadata == "N5Y":
            seq = "ABCDYDAC"
        else:
            assert(ret0.metadata == "N5Y,D6G")
            seq = "ABCDYGAC"

        tokenizations = get_tokenizations_for_sequence(seq)

        assert(check_tensor_against_list(ret0.itokens, tokenizations[0][0])), "%s =/= %s" % (str(ret0.itokens), str(tokenizations[0][0]))
        assert(check_tensor_against_list(ret0.otokens, tokenizations[0][1])), "%s =/= %s" % (str(ret0.otokens), str(tokenizations[0][1]))
        assert(check_tensor_against_list(ret1.itokens, tokenizations[1][0])), "%s =/= %s" % (str(ret1.itokens), str(tokenizations[1][0]))
        assert(check_tensor_against_list(ret1.otokens, tokenizations[1][1])), "%s =/= %s" % (str(ret1.otokens), str(tokenizations[1][1]))


def test_sars_cov2_data_processing():
    tsvname, ref_file = create_data()
    args = argparse.Namespace(
        tsv=tsvname,
        last_train_date="2019-12-15",
        last_val_date="2019-12-16",
        protein_name="Spike",
        frac_train=0.9,
        min_val_pangos=1,
        datadir=os.path.join(testdir, "data"),
        min_masked_segment=0,
        max_masked_segment=5,
        randint_functor=lambda *args, **kwargs: 5,
        randsample_functor=lambda *args, **kwargs: 5,
        ref_file=ref_file,
        num_randomizations_per_sequence_train=3,
        num_randomizations_per_sequence_val=2,
        min_random_segments_per_seq=4,
        select_segments=False,
        sort_field="Collection date",
        reconstruction_filter=False,
    )
    sars_cov2_data_processing.main(args)
    train_data = data_processing.CompoundDiskStorageReader(
        os.path.join(testdir, "data"), data_type="train",
    )
    validate_dset(train_data, 3)
    val_data = data_processing.CompoundDiskStorageReader(
        os.path.join(testdir, "data"), data_type="val",
    )
    validate_dset(val_data, 2)
    all_metadata = set([train_data[0].metadata, val_data[0].metadata])
    assert(all_metadata == {"N5Y", "N5Y,D6G"})
    print("Test test_sars_cov2_data_processing passed")


def validate_dset_select_segments(
    dset: data_processing.CompoundDiskStorageReader,
) -> None:
    if dset[0].metadata == "ins1A,N5del,D6del,A7del":
        assert(len(dset) == 3)
        seq = "AABCDC"
        tokenizations = get_tokenizations_for_sequence(seq)

        ret0 = dset[0]
        ret0.tensorize()
        assert(check_tensor_against_list(ret0.itokens, tokenizations[0][0])), "%s =/= %s" % (str(ret0.itokens), str(tokenizations[0][0]))
        assert(check_tensor_against_list(ret0.otokens, tokenizations[0][1])), "%s =/= %s" % (str(ret0.otokens), str(tokenizations[0][1]))

        ret1 = dset[1]
        ret1.tensorize()
        assert(check_tensor_against_list(ret1.itokens, tokenizations[1][0])), "%s =/= %s" % (str(ret1.itokens), str(tokenizations[0][0]))
        assert(check_tensor_against_list(ret1.otokens, tokenizations[1][1])), "%s =/= %s" % (str(ret1.otokens), str(tokenizations[0][1]))

        ret2 = dset[2]
        ret2.tensorize()
        assert(check_tensor_against_list(ret2.itokens, tokenizations[0][0])), "%s =/= %s" % (str(ret2.itokens), str(tokenizations[0][0]))
        assert(check_tensor_against_list(ret2.otokens, tokenizations[0][1])), "%s =/= %s" % (str(ret2.otokens), str(tokenizations[0][1]))
    elif dset[0].metadata == "N5Y,D6G,A7G":
        assert(len(dset) == 3)
        seq = "ABCDYGGC"
        tokenizations = get_tokenizations_for_sequence(seq)

        ret0 = dset[0]
        ret0.tensorize()
        assert(check_tensor_against_list(ret0.itokens, tokenizations[0][0])), "%s =/= %s" % (str(ret0.itokens), str(tokenizations[0][0]))
        assert(check_tensor_against_list(ret0.otokens, tokenizations[0][1])), "%s =/= %s" % (str(ret0.otokens), str(tokenizations[0][1]))

        ret1 = dset[1]
        ret1.tensorize()
        assert(check_tensor_against_list(ret1.itokens, tokenizations[1][0])), "%s =/= %s" % (str(ret1.itokens), str(tokenizations[1][0]))
        assert(check_tensor_against_list(ret1.otokens, tokenizations[1][1])), "%s =/= %s" % (str(ret1.otokens), str(tokenizations[1][1]))

        ret2 = dset[2]
        ret2.tensorize()
        assert(check_tensor_against_list(ret2.itokens, tokenizations[0][0])), "%s =/= %s" % (str(ret2.itokens), str(tokenizations[0][0]))
        assert(check_tensor_against_list(ret2.otokens, tokenizations[0][1])), "%s =/= %s" % (str(ret2.otokens), str(tokenizations[0][1]))
    else:
        raise ValueError(f"Unknown sequence: {dset[0].metadata}")


def test_sars_cov2_data_processing_select_segments():
    tsvname, ref_file = create_data_select_segments()
    args = argparse.Namespace(
        tsv=tsvname,
        last_train_date="2019-12-15",
        last_val_date="2019-12-16",
        protein_name="Spike",
        frac_train=0.9,
        min_val_pangos=1,
        datadir=os.path.join(testdir, "data"),
        min_masked_segment=0,
        max_masked_segment=5,
        randint_functor=lambda *args, **kwargs: 5,
        randsample_functor=lambda *args, **kwargs: [args[0][0]],
        ref_file=ref_file,
        num_randomizations_per_sequence_train=1,
        num_randomizations_per_sequence_val=1,
        min_random_segments_per_seq=1,
        select_segments=True,
        sort_field="Collection date",
        reconstruction_filter=False,
    )
    sars_cov2_data_processing.main(args)
    train_data = data_processing.CompoundDiskStorageReader(
        os.path.join(testdir, "data"), data_type="train",
    )
    validate_dset_select_segments(train_data)
    val_data = data_processing.CompoundDiskStorageReader(
        os.path.join(testdir, "data"), data_type="val",
    )
    validate_dset_select_segments(val_data)
    all_metadata = set([
        train_data[i].metadata for i in range(len(train_data))
    ] + [val_data[i].metadata for i in range(len(val_data))])
    assert(all_metadata == {"ins1A,N5del,D6del,A7del", "N5Y,D6G,A7G"})
    print("Test test_sars_cov2_data_processing_select_segments passed")


def test_mutation_positions_in_seq():
    reference = "AYBG--ATGN-A".replace("-", "")
    sequence  = "A--GABAG-NAA".replace("-", "")
    mutations = "Y2del,B3del,ins4AB,T6G,G7del,ins8A"
    res = sars_cov2_data_processing.mutation_positions_in_seq(mutations)
    expected = [(0, 2), (1, 5), (5, 6), (5, 7), (6, 9)]
    assert(expected == res), f"Expected={expected}, res={res}"
    print("Test test_mutation_positions_in_seq passed")


if __name__ == "__main__":
    test_sars_cov2_data_processing()
    test_sars_cov2_data_processing_select_segments()
    test_mutation_positions_in_seq()
