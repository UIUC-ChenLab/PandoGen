# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pysam
import json
import argparse


def main(args):
    with open(args.ref) as fhandle:
        ref = fhandle.read().strip()

    with open(args.predictions, "r") as fhandle, open(args.output_file, "w") as whandle:
        for item in fhandle:
            seq = json.loads(item)
            if "[" in seq["seq"] or "]" in seq["seq"]:
                continue
            res = {
                "orig_sequence": ref,
                "generations": [
                    [],
                    [
                        {"parent": ref, "child": seq["seq"], "ll": seq["ll"]}
                    ]
                ]
            }
            whandle.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert decoder output to generations output")

    parser.add_argument("--predictions", help="Decoder predictions", required=True)
    parser.add_argument("--output_file", help="Output file name", required=True)
    parser.add_argument("--ref", help="Reference sequence", required=True)

    args = parser.parse_args()
    main(args)
