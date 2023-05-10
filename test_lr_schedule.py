# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
"""
Test Learning Rate scheduling with parameter groups
"""
import torch
from transformers.optimization import get_linear_schedule_with_warmup
import train
import copy


def print_scheduling():
    class SomeNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.Tensor(4, 3))
            self.b = torch.nn.Parameter(torch.Tensor(3, 4))

        def forward(self):
            return torch.sum(torch.matmul(self.a, self.b) ** 2)


    model = SomeNetwork()
    optimizer = torch.optim.AdamW([{"params": model.a, "lr": 1e-3}, {"params": model.b, "lr": 5e-2}], weight_decay=1e-2)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=20)

    for i in range(21):
        print("lr_a", optimizer.param_groups[0]["lr"], i)
        print("lr_b", optimizer.param_groups[1]["lr"], i)
        optimizer.step()
        scheduler.step()


def test_variable_grouping():
    parameter_names = [
        "encoder.embeddings.word_embeddings.weight",
        "encoder.embeddings.position_embeddings.weight",
        "encoder.embeddings.LayerNorm.weight",
        "encoder.encoder.layer.0.attention.self.query.weight",
        "encoder.encoder.layer.0.attention.self.query.bias",
        "encoder.encoder.layer.1.attention.self.key.weight",
        "encoder.encoder.layer.1.attention.self.key.bias",
        "decoder.cls.predictions.bias",
        "decoder.bert.encoder.layer.7.output.dense.weight",
        "decoder.bert.encoder.layer.7.output.dense.bias",
        "decoder.bert.encoder.layer.7.output.LayerNorm.weight",
        "decoder.cls.predictions.transform.dense.weight",
    ]

    parameter_name_groupings = {
        "non_attn_parameters": {
            "decay_parameters": [
                "encoder.embeddings.position_embeddings.weight",
                "decoder.cls.predictions.transform.dense.weight",
            ],
            "nondecay_parameters": [
                "decoder.cls.predictions.bias",
            ]
        },
        "attn_parameters": {
            "encoder_layers": {
                0: {
                    "decay_parameters": [
                        "encoder.embeddings.word_embeddings.weight",
                        "encoder.encoder.layer.0.attention.self.query.weight",
                    ],
                    "nondecay_parameters": [
                        "encoder.embeddings.LayerNorm.weight",
                        "encoder.encoder.layer.0.attention.self.query.bias",
                    ],
                },
                1: {
                    "decay_parameters": [
                        "encoder.encoder.layer.1.attention.self.key.weight",
                    ],
                    "nondecay_parameters": [
                        "encoder.encoder.layer.1.attention.self.key.bias",
                    ]
                }
            },
            "decoder_layers": {
                7: {
                    "decay_parameters": [
                        "decoder.bert.encoder.layer.7.output.dense.weight",
                    ],
                    "nondecay_parameters": [
                        "decoder.bert.encoder.layer.7.output.dense.bias",
                        "decoder.bert.encoder.layer.7.output.LayerNorm.weight",
                    ]
                }
            }
        }
    }

    class Model:
        def __init__(self):
            self.parameter_maps = {
                key: torch.LongTensor([i]) for i, key in enumerate(parameter_names)
            }

        def named_parameters(self):
            return list(self.parameter_maps.items())

    def get_parameter_names(model, *args, **kwargs):
        for key, value in model.named_parameters():
            if "LayerNorm" in key:
                continue
            yield key

    train.get_parameter_names = get_parameter_names

    model = Model()

    parameter_groupings = copy.deepcopy(parameter_name_groupings)

    def recursive_assign(d: dict):
        if "decay_parameters" in d:
            d["decay_parameters"] = [
                model.parameter_maps[k] for k in d["decay_parameters"]
            ]
            d["nondecay_parameters"] = [
                model.parameter_maps[k] for k in d["nondecay_parameters"]
            ]
        else:
            for key in d:
                recursive_assign(d[key])

    recursive_assign(parameter_groupings)

    lr = 1
    wd = 0.1

    parameter_groups_for_optim = [
        {"params": [model.parameter_maps["decoder.bert.encoder.layer.7.output.dense.weight"]], "lr": lr, "weight_decay": wd},
        {"params": [model.parameter_maps["decoder.bert.encoder.layer.7.output.dense.bias"],
            model.parameter_maps["decoder.bert.encoder.layer.7.output.LayerNorm.weight"]], "lr": lr},
        {"params": [model.parameter_maps["encoder.encoder.layer.1.attention.self.key.weight"]],
            "lr": lr, "weight_decay": wd},
        {"params": [model.parameter_maps["encoder.encoder.layer.1.attention.self.key.bias"]],
            "lr": lr},
        {"params": [model.parameter_maps["encoder.embeddings.word_embeddings.weight"],
            model.parameter_maps["encoder.encoder.layer.0.attention.self.query.weight"]], "lr": lr * 0.5, "weight_decay": wd},
        {"params": [model.parameter_maps["encoder.embeddings.LayerNorm.weight"],
            model.parameter_maps["encoder.encoder.layer.0.attention.self.query.bias"]], "lr": lr * 0.5},
        {"params": [model.parameter_maps["encoder.embeddings.position_embeddings.weight"],
            model.parameter_maps["decoder.cls.predictions.transform.dense.weight"]], "lr": lr, "weight_decay": wd},
        {"params": [model.parameter_maps["decoder.cls.predictions.bias"]], "lr": lr},
    ]

    obt_groupings = train.get_layerwise_groupings(model)
    obt_groupings = {
        "non_attn_parameters": obt_groupings["non_attn_parameters"],
        "attn_parameters": {
            "encoder_layers": {k: dict(v) for k, v in obt_groupings["attn_parameters"]["encoder_layers"].items()},
            "decoder_layers": {k: dict(v) for k, v in obt_groupings["attn_parameters"]["decoder_layers"].items()},
        }
    }
    assert(obt_groupings == parameter_groupings), f"{obt_groupings} != {parameter_groupings}"

    optimizer = train.get_finetuning_optimizer(
        model,
        lr=lr,
        lr_decay_rate=0.5,
        weight_decay=wd,
        optimizer_kwargs={},
    )

    exp_optimizer = torch.optim.AdamW(parameter_groups_for_optim)

    a = sorted(optimizer.param_groups, key=lambda x: tuple(x.items()))
    b = sorted(exp_optimizer.param_groups, key=lambda x: tuple(x.items()))

    assert(a == b), f"{a} != {b}"

    print("Test test_variable_grouping passed")


if __name__ == "__main__":
    print_scheduling()
    test_variable_grouping()
