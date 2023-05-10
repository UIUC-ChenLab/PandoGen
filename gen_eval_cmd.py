# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import os
import sys
import re
import subprocess
import logging
from argparse import Namespace, ArgumentParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


def create_checkpoint_eval_cmd(
    checkpoint: str,
    results_dir: str,
    batch_size: int,
    dataset: str,
) -> list:
    results_subst = os.path.join(
        os.path.split(os.path.split(checkpoint)[0])[1],
        os.path.split(checkpoint)[1],
    )

    checkpoint_eval_results = os.path.join(results_dir, results_subst)

    if os.path.exists(
        os.path.join(checkpoint_eval_results, "eval_results.json")
    ):
        return

    eval_cmd_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "eval.py")

    cmd = [
        "python",
        eval_cmd_path,
        "--distributed_dataset", dataset,
        "--output_dir", checkpoint_eval_results,
        "--do_eval",
        "--per_device_eval_batch_size", str(batch_size),
        "--dataloader_num_workers", "4",
        "--checkpoint_params", checkpoint,
    ]

    return cmd
    

def main(args: Namespace) -> None:
    checkpoints = []

    for checkpoint in os.listdir(args.directory):
        if not checkpoint.startswith("checkpoint"):
            continue

        logging.info("Running evaluation for checkpoint")

        checkpoint = os.path.join(args.directory, checkpoint)
        command = create_checkpoint_eval_cmd(
            checkpoint, args.results_dir, args.batch_size, args.dataset)

        if not command:
            logging.info("Checkpoint already evaluated. Skipping.")
            continue

        logging.info(f"Running evaluation command {command}")

        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Perform evaluations for a directory of checkpoints")
    parser.add_argument("--directory", help="Directory with checkpoints", required=True)
    parser.add_argument("--results_dir", help="Output directory for eval logs", required=True)
    parser.add_argument("--batch_size", help="Batch size to use", default=64, type=int)
    parser.add_argument("--dataset", help="Dataset for eval", required=True)
    parser.add_argument("--dry_run", help="Display commands, don't run", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
