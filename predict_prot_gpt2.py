# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import re
import torch
from argparse import ArgumentParser
from prediction_tools import add_sampler_parameters
import transformers
import json
from prediction_tools import calc_likelihoods
from quark_finetune import calc_likelihoods_top


def clean_batch(batch: list):
    for item in batch:
        item = item[len("<|endoftext|>"): ]
        item = re.sub(r"\*+$", "*", item)
        yield item


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    gen_kwargs = {a[4:]: getattr(args, a) for a in vars(args) if a[:4] == "gen_"}
    eos_token = tokenizer("*")["input_ids"][0]
    results = []

    for i in tqdm.tqdm(range(args.n_batches), desc="Generating"):
        batch = model.generate(
            **gen_kwargs,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=eos_token,
            pad_token_id=eos_token,
        )
        with torch.no_grad():
            lls = calc_likelihoods_top(model, res=batch, is_quark_model=False, eos_token=eos_token, temperature=args.gen_temperature).cpu().tolist()

        decoded = tokenizer.batch_decode(batch)
        clean = list(clean_batch(decoded))
        for seq, l in zip(clean, lls):
            results.append({"seq": seq, "ll": l})

    with open(args.output_prefix + ".json", "w") as fhandle:
        for item in results:
            fhandle.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate ProtGPT2 sequences")    

    parser.add_argument("--checkpoint", help="ProtGPT2 checkpoint", required=True)
    parser.add_argument("--n_batches", help="Batch size", required=True, type=int)
    parser.add_argument("--seed", help="Batch size", required=False, type=int)
    parser.add_argument("--output_prefix", help="Prefix of output file", required=True)

    add_sampler_parameters(parser)

    args = parser.parse_args()

    if args.seed:
        transformers.set_seed(args.seed)

    main(args)
