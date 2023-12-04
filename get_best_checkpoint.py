# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import argparse
import os
import json


def get_eval_loss(ckpt: str) -> float:
    with open(os.path.join(ckpt, "trainer_state.json")) as fhandle:
        ckpt_stats = json.load(fhandle)
    
    ckpt_loss = ckpt_stats["log_history"][-1]["eval_loss"]

    return ckpt_loss


def main(args):
    directories = [
        os.path.join(args.path, ckpt) for ckpt in os.listdir(args.path)
        if ckpt.startswith("checkpoint")
    ]

    selector = max if args.greater_is_better else min

    best_ckpt = selector(directories, key=get_eval_loss)

    print(f"Best checkpoint is {best_ckpt}")

    if args.cleanup_command:
        with open(args.cleanup_command, "w") as fhandle:
            fhandle.write("#!/bin/bash\n")
            for ckpt in directories:
                cmd = f"rm -r {ckpt}"
                if ckpt == best_ckpt:
                    cmd = f"# best checkpoint -> {cmd}"
                fhandle.write(cmd + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the checkpoint with the best eval loss"
    )

    parser.add_argument(
        "--path",
        help="Base path where checkpoints may be found",
        required=True,
    )

    parser.add_argument(
        "--greater_is_better",
        help="Loss is better if greater",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--cleanup_command",
        help="Create a shell script for cleanup",
        required=False,
    )

    args = parser.parse_args()

    main(args)
