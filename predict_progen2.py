# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import sys
import transformers
from dataclasses import dataclass, field, replace
import torch
from tokenizers import Tokenizer
import logging
import os
import json
import torch.distributed
import hashlib


@dataclass
class Arguments:
    progen2_src: str = field(
        metadata={"help": "Path of progen2"}
    )

    progen2_ckpt: str = field(
        metadata={"help": "Progen2 checkpoint"}
    )

    total_to_generate: int = field(
        metadata={"help": "Number of sequences to generate"}
    )

    gen_top_p: float = field(
        default=None,
        metadata={"help": "Generation top_p value"}
    )

    gen_temperature: float = field(
        default=None,
        metadata={"help": "Temperature for generation"}
    )

    max_prediction_length: int = field(
        default=1400,
        metadata={"help": "How long of a sequence to generate"}
    )

    output_filename: str = field(
        default="predictions.fa",
        metadata={"help": "Name of the output file"}
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_sequences: int, tokenizer_path: str):
        super().__init__()
        self._length = n_sequences
        with open(os.path.join(tokenizer_path, "progen2", "tokenizer.json"), "r") as fhandle:
            self.tokenizer = Tokenizer.from_str(fhandle.read())

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.tokenizer.encode("1").ids}


def main(args: Arguments, s2sargs: transformers.Seq2SeqTrainingArguments):
    if s2sargs.world_size > 1:
        seed_string = f"{s2sargs.seed}{s2sargs.world_size}{s2sargs.process_index}"
        m = hashlib.sha256()
        m.update(seed_string.encode("utf-8"))
        seed = int(m.hexdigest()[:8], 16)
        logging.info(f"For process {s2sargs.process_index}, seed={seed}")
    else:
        seed = s2sargs.seed

    # seeding is redone in Trainer:
    # https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/trainer.py#L342
    # So change the individual process's seed in the argument
    # training_args seems to be frozen in some transformers versions. Hence unfreezing.
    # s2sargs.seed = seed
    s2sargs = replace(s2sargs, seed=seed)
    transformers.set_seed(seed)

    # Setting log-levels
    log_level = s2sargs.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Import progen code
    sys.path.append(args.progen2_src)
    import progen2.models.progen.modeling_progen as modeling_progen

    if s2sargs.world_size == 1 or s2sargs.process_index == 0:
        logging.info("Initializing placeholder dataset")

    dataset = Dataset(args.total_to_generate, tokenizer_path=args.progen2_src)

    ckpt = args.progen2_ckpt

    if s2sargs.world_size > 1:
        if s2sargs.process_index == 0:
            logging.info("Multi-GPU run: Invoking barrier")
        torch.distributed.barrier()

    if s2sargs.world_size == 1 or s2sargs.process_index == 0:
        logging.info("Loading model")

    if s2sargs.fp16:
        if s2sargs.deepspeed is None:
            model = modeling_progen.ProGenForCausalLM.from_pretrained(
                ckpt, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True,
            )
        else:
            model = modeling_progen.ProGenForCausalLM.from_pretrained(
                ckpt, revision="float16", torch_dtype=torch.float16,
            )
    else:
        model = modeling_progen.ProGenForCausalLM.from_pretrained(ckpt)

    gen_config = transformers.GenerationConfig(
        max_new_tokens=args.max_prediction_length,
        do_sample=True,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        pad_token_id=dataset.tokenizer.encode("<|pad|>").ids[0],
        eos_token_id=dataset.tokenizer.encode("2").ids[0],
        num_return_sequences=1,
    )

    # s2sargs.generation_config = gen_config
    # s2sargs.predict_with_generate = True
    s2sargs = replace(s2sargs, generation_config=gen_config, predict_with_generate=True)

    generator = transformers.Seq2SeqTrainer(
        model=model,
        args=s2sargs,
    )

    results = generator.predict(dataset)

    if s2sargs.world_size == 1 or s2sargs.process_index == 0:
        with open(os.path.join(s2sargs.output_dir, args.output_filename), "w") as fhandle:
            for i, r in enumerate(results.predictions.tolist()):
                decoded = dataset.tokenizer.decode(r)
                if decoded[0] == "1":
                    decoded = decoded[1:]
                decoded = decoded.replace("2", "*")
                fhandle.write(f">{i}\n{decoded}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = transformers.HfArgumentParser((Arguments, transformers.Seq2SeqTrainingArguments))
    args, s2sargs = parser.parse_args_into_dataclasses()
    main(args, s2sargs)
