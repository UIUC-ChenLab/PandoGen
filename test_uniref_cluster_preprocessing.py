# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import uniref_cluster_preprocessing
from argparse import Namespace
import os
import pickle
import sys
import re
import shutil


def ad_hoc_uniref_processing(filename: str, entry_ids: list) -> dict:
    counter = 0
    ad_hoc_uniref100_dict = {}
    ad_hoc_uniref90_dict = {}

    with open(filename, "r") as fhandle:
        while True:
            line = next(fhandle).strip()

            if line.startswith("<entry id="):
                uniref_id = re.findall(r"<entry id=\"(\S+)\"\s*updated=\"\S+\">", line)[0]

            if line.startswith("<representativeMember") or line.startswith("<member"):
                member_type = line.split()[0][1:]
                member_dict = {}

                while True:
                    line = next(fhandle).strip()

                    if line in ["</representativeMember>", "</member>"]:
                        break

                    property_tag = re.findall(r"<property type=\"(.*)\"\s*value=\"(.*)\"/>", line)

                    if counter + 1 in entry_ids:
                        if property_tag:
                            property_tag = property_tag.pop()
                            if property_tag[0] == "UniRef100 ID":
                                ad_hoc_uniref100_dict[property_tag[1]] = uniref_id

                            if property_tag[0] == "UniRef90 ID":
                                ad_hoc_uniref90_dict[property_tag[1]] = uniref_id

            if line == "</entry>":
                counter += 1

            if counter > max(entry_ids):
                break

    return ad_hoc_uniref100_dict, ad_hoc_uniref90_dict


def load_pkl(pkl_name: str) -> dict:
    with open(pkl_name, "rb") as fhandle:
        return pickle.load(fhandle)


def test_uniref50_cluster(xml_name: str, xsd_name: str):
    test_prefix = "/tmp/test_uniref50_cluster"
    uniref100_res = f"{test_prefix}_3.uniref100.pkl"
    uniref90_res = f"{test_prefix}_3.uniref90.pkl"

    if os.path.exists(uniref100_res):
        os.remove(uniref100_res)

    if os.path.exists(uniref90_res):
        os.remove(uniref90_res)

    args = Namespace(
        xml=xml_name,
        schema=xsd_name,
        output_prefix=test_prefix,
        local_rank=3,
        world_size=24,
        max_to_read=3,
    )

    uniref_cluster_preprocessing.main(args)

    uniref100_results = load_pkl(uniref100_res)
    uniref90_results = load_pkl(uniref90_res)

    uniref100_exp_results, uniref90_exp_results = ad_hoc_uniref_processing(
        xml_name, entry_ids=[4, 28, 52]
    )

    assert(uniref100_results == uniref100_exp_results), f"{uniref100_results} != {uniref100_exp_results}"
    assert(uniref90_results == uniref90_exp_results), f"{uniref90_results} != {uniref90_exp_results}"

    print("Test test_uniref50_cluster passed")


def test_train_val_splitter():
    testfiles = ["/tmp/clusters/cluster-0.pkl", "/tmp/clusters/cluster-1.pkl"]

    # Test 1
    dicts = {"A": "abc", "B": "xyz", "C": "xyz", "D": "abc"}
    splitter = uniref_cluster_preprocessing.TrainValSplitter(dicts)
    A = splitter([1, 2], weights=[0.5, 0.5], k=1, uniref_id="A")[0]
    B = splitter([1, 2], weights=[0.5, 0.5], k=1, uniref_id="B")[0]
    C = splitter([1, 2], weights=[0.5, 0.5], k=1, uniref_id="C")[0]
    D = splitter([1, 2], weights=[0.5, 0.5], k=1, uniref_id="D")[0]
    # Hash values calculated manually (and truncated to 10k) are
    # abc=1821, xyz=8994
    assert(A == D == 1 and B == C == 2)

    # Test 2
    dicts = {"A": "a", "B": "b", "C": "c", "D": "d"}
    splitter = uniref_cluster_preprocessing.TrainValSplitter(dicts)
    A = splitter([1, 2], weights=[0.5, 0.5], k=1, uniref_id="A")[0]
    B = splitter([1, 2], weights=[0.5, 0.5], k=1, uniref_id="B")[0]
    C = splitter([1, 2], weights=[0.5, 0.5], k=1, uniref_id="C")[0]
    D = splitter([1, 2], weights=[0.5, 0.5], k=1, uniref_id="D")[0]

    # Hash values calculated manually (and truncated to 10k) are
    # a=8987, b=781, c=8614, d=9732
    # For 10,000 buckets, a, c, d fall in the second half
    # and b falls in the first half
    assert(A == C == D == 2 and B == 1)

    print("Test test_train_val_splitter passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        test_train_val_splitter()
        print("Usage: python test_uniref_cluster_preprocessing.py <uniref50 xml> <uniref xsd>")
    else:
        test_uniref50_cluster(sys.argv[1], sys.argv[2])
        test_train_val_splitter()
