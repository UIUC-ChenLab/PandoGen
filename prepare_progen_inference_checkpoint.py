# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from dataclasses import dataclass, field
import transformers
from transformers import HfArgumentParser
from tokenizers import Tokenizer
import os
import logging
import shutil
import json


@dataclass
class Arguments:
    progen2_src: str = field(
        metadata={"help": "Path of progen2"}
    )

    progen2_ckpt: str = field(
        metadata={"help": "Progen2 checkpoint"}
    )

    n_positions: int = field(
        metadata={"help": "Number of positions to generate for"}
    )

    output_path: str = field(
        metadata={"help": "Output checkpoint path"}
    )


def fix_ckpt(ckpt: str, workdir: str, pad_token_id, max_length: int = 2048, process_id: int = None, no_overwrite: bool = False) -> str:
    """
    Fix n_positions in model checkpoint
    """
    gen_ckpt = os.path.join(workdir, "progen_checkpoint_for_gen")

    if not process_id:
        logging.info("Fixing checkpoint")

        if not os.path.exists(gen_ckpt):
            shutil.copytree(ckpt, gen_ckpt)
        else:
            if not no_overwrite:
                raise ValueError(f"{gen_ckpt} exists, but overwrite requested!")

        with open(os.path.join(gen_ckpt, "config.json"), "r") as fhandle:
            config = json.load(fhandle)

        config["n_positions"] = max_length
        config["pad_token_id"] = pad_token_id

        with open(os.path.join(gen_ckpt, "config.json"), "w") as fhandle:
            json.dump(config, fhandle, indent=4)    

    return gen_ckpt


def main(args: Arguments):
    with open(os.path.join(args.progen2_src, "progen2", "tokenizer.json"), "r") as fhandle:
        tokenizer = Tokenizer.from_str(fhandle.read())

    fixed_ckpt = fix_ckpt(
        ckpt=args.progen2_ckpt,
        workdir=args.output_path,
        pad_token_id=tokenizer.encode("<|pad|>").ids[0],
        max_length=args.n_positions,
        process_id=None,
        no_overwrite=False,
    )

    logging.info(f"Fixed checkpoint = {fixed_ckpt}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = HfArgumentParser((Arguments, ))
    args, = parser.parse_args_into_dataclasses()
    main(args)
