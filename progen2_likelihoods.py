# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import sys
import transformers
from dataclasses import dataclass, field
import torch
from tokenizers import Tokenizer
import logging
import os
import json
import torch.distributed
import hashlib
import utils
import torch
from prediction_tools import calc_likelihoods
from functools import partial
import tqdm


@dataclass
class Arguments:
    fa: str = field(
        metadata={"help": "Predicted fasta file"}
    )

    progen_src: str = field(
        metadata={"help": "ProGen2 source file"}
    )

    progen_checkpoint: str = field(
        metadata={"help": "Checkpoint path"}
    )

    output_prefix: str = field(
        metadata={"help": "Prefix of output json file"}
    )

    fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16"}
    )

    temperature: float = field(
        default=1,
        metadata={"help": "Temperature value for generation"}
    )

    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for likelihood calculations"}
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, fasta_file: str, progen_src: str):
        super().__init__()
        with open(os.path.join(progen_src, "progen2", "tokenizer.json"), "r") as fhandle:
            self.tokenizer = Tokenizer.from_str(fhandle.read())
        self.records = [
            f"1{x.sequence}".replace("*", "2") for x in utils.fasta_serial_reader(fasta_file)
            if x.sequence[-1] == "*" and x.sequence.count("*") == 1
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        seq = self.records[idx]
        tokenized = self.tokenizer.encode(seq)
        return torch.LongTensor(tokenized.ids), seq


def collator(batch: list, pad_value: int):
    batch_size = len(batch)
    tensors, sequences = tuple(zip(*batch))
    max_length = max(i.shape[0] for i in tensors)
    input_ids = torch.full([batch_size, max_length], fill_value=pad_value).long()

    for i, b in enumerate(tensors):
        input_ids[i, :b.shape[0]] = b

    return {"input_ids": input_ids}, sequences


def main(args: Arguments):
    sys.path.append(args.progen_src)
    import progen2.models.progen.modeling_progen as modeling_progen

    ckpt = args.progen_checkpoint

    if args.fp16:
        model = modeling_progen.ProGenForCausalLM.from_pretrained(
            ckpt, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )
    else:
        model = modeling_progen.ProGenForCausalLM.from_pretrained(ckpt)

    model.eval()
    model.cuda()

    data = Dataset(args.fa, args.progen_src)

    loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=torch.utils.data.SequentialSampler(data),
        pin_memory=True,
        num_workers=0,
        drop_last=False,
        collate_fn=partial(collator, pad_value=data.tokenizer.encode("<|pad|>").ids[0]),
    )

    eos_token_id = data.tokenizer.encode("2").ids[0]

    results = []

    for i, batch in enumerate(tqdm.tqdm(loader, desc="Calculating likelihoods")):
        input_data, sequences = batch

        input_data = {key: value.cuda() for key, value in input_data.items()}

        with torch.no_grad():
            outputs = model(**input_data)

        logits = outputs.logits
        seq = input_data.get("input_ids")[:, 1: ]
        seq_logits = logits[:, : -1]

        if args.temperature:
            seq_logits = seq_logits / args.temperature

        likelihoods = calc_likelihoods(
            seq,
            end_token_value=eos_token_id,
            logits=seq_logits,
        ).cpu().tolist()

        results.extend(
            zip(sequences, likelihoods)
        )

    with open(args.output_prefix + ".json", "w") as fhandle:
        for seq, l in results:
            result_dict = {
                "seq": seq[1:].replace("2", "*"),
                "ll": l
            }
            fhandle.write(json.dumps(result_dict) + "\n")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((Arguments, ))
    args, = parser.parse_args_into_dataclasses()
    main(args)
