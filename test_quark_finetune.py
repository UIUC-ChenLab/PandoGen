# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import quark_finetune
import torch
import data_processing
from argparse import Namespace
import os
import shutil
import json


def to_list_helper(data: list) -> dict:
    return [{
        key: value.tolist() if isinstance(value, torch.Tensor) else value for \
            key, value in d.items()
    } for d in data]


def test_pad_equally():
    seq_a = torch.Tensor([
        [1, 1, 1, 0],
        [2, 2, 2, 2],
    ])

    seq_b = torch.Tensor([
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 0, 0]
    ])

    res_a, res_b = quark_finetune.pad_equally(seq_a, seq_b)

    exp_a = torch.Tensor([
        [1, 1, 1, 0, 0, 0],
        [2, 2, 2, 2, 0, 0]
    ])

    assert(torch.all(res_a == exp_a))
    assert(torch.all(res_b == seq_b))

    print("Test test_pad_equally passed")


def test_reward_model():
    class DummyScorer:
        def __init__(self):
            pass

        def __call__(self, *args, **kwargs):
            return torch.Tensor([
                1.5,
                0.5,
                0.25
            ])

    prev_sequences = torch.ByteTensor(
        [
            [1, 2, 3, 4, 5, 6, 9],
            [3, 4, 5, 6, 9, 0, 0],
        ]
    )

    reward_model = quark_finetune.RewardModel(
        DummyScorer(),
        eos_token=9,
        prev_sequences=prev_sequences,
    )

    test_sequences = {
        "input_ids": torch.ByteTensor([
            [26, 1, 2, 3, 4, 5, 6, 9, -1, -2],
            [26, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [26, 5, 4, 3, 2, 1, 9, 0, 0, 0],
        ]),
        "attention_mask": torch.ByteTensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        ]),
    }

    potentials, membership, sequence_valid = reward_model(**test_sequences)

    assert(potentials.cpu().tolist() == [1.5, 0.5, 0.25])
    assert(membership.cpu().tolist() == [1, 0, 0]), f"{membership} != {[1, 0, 0]}"
    assert(sequence_valid.cpu().tolist() == [1, 0, 1])

    print("Test test_reward_model passed")


def test_quark_forward():
    class DummyModelRef:
        def __init__(self):
            pass

        def __call__(self, *args, **kwargs):
            return Namespace(
                logits=torch.Tensor(
                    [
                        [[9, -9], [2, 2], [-1, 1]],
                        [[0, 3], [2, 1], [0, 0]]
                    ]
                )
            )

    class DummyModelTrain:
        def __init__(self):
            pass

        def __call__(self, *args, **kwargs):
            return Namespace(
                logits=torch.Tensor(
                    [
                        [[1, 1], [9, -9], [2, 2], [-1, 1]],
                        [[1, 0], [0, 3], [2, 1], [0, 0]]
                    ]
                )
            )

    train_sequences = {
        "input_ids": torch.LongTensor(
            [
                [26, 30, 0, 1],
                [26, 30, 1, 0]
            ]
        ),
        "attention_mask": torch.ByteTensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ]
        )
    }

    ref_sequences = {
        "input_ids": torch.LongTensor(
            [
                [26, 0, 1],
                [26, 1, 0]
            ]
        ),
        "attention_mask": torch.ByteTensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ]
        )
    }

    model = Namespace(
        ref_model=DummyModelRef(),
        train_model=DummyModelTrain(),
    )

    res = quark_finetune.quark_forward(
        model,
        train_sequences,
        ref_sequences,
    )

    def get_softmax(l: list) -> torch.Tensor:
        ll = torch.log_softmax(torch.Tensor(l), dim=-1)
        return ll

    model_loss = -(
        get_softmax([9, -9])[0] + get_softmax([2, 2])[1] + get_softmax([0, 3])[1]) / 3

    def float_eq(x, y, eps=1e-4):
        return x.item() - eps <= y.item() <= x.item() + eps

    assert(float_eq(model_loss, res[0])), f"{model_loss.item()} != {res[0].item()}"
    assert(float_eq(res[1], torch.Tensor([0.0])[0])), f"{res[1].item()} != 0.0"

    print("Test test_quark_forward passed")


def test_get_attn_mask():
    sequence = torch.LongTensor(
        [
            [1, 2, 3, 4, 5, 9, 10, 11],
            [3, 4, 5, 9, 9, 9, 0, 0]
        ]
    )
    attn_mask = quark_finetune.get_attn_mask(sequence, eos_token=9)
    exp_attn_mask = torch.BoolTensor(
        [
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False, False],
        ]
    )
    assert(torch.all(attn_mask == exp_attn_mask)), f"{attn_mask} != {exp_attn_mask}"
    print("Test test_get_attn_mask passed")


def test_generate_datapool_batch():
    class DummyModel():
        def __init__(self, *args, **kwargs):
            pass

        def eval(self, *args, **kwargs):
            pass

        def generate(self, input_ids = None, bos_token_id = None, eos_token_id = None, gen_kwargs: dict = None):
            if bos_token_id is not None:
                bos = [bos_token_id]
            else:
                bos = input_ids.tolist()[0]

            return torch.LongTensor(
                [
                    bos + [1, 2, 3, 4, 5, 6, eos_token_id, -1, -2],
                    bos + [1, 2, 3, 4, 5, 6, 7, 8, 10],
                    bos + [5, 4, 3, 2, 1, eos_token_id, 0, 0, 0],
                ]
            )

    class DummyScorer:
        def __init__(self):
            pass

        def __call__(self, *args, **kwargs):
            return torch.Tensor([
                1.5,
                0.5,
                0.25
            ])[:, None]

    prev_sequences = torch.ByteTensor(
        [
            [1, 2, 3, 4, 5, 6, 9],
            [3, 4, 5, 6, 9, 0, 0],
        ]
    )
    eos_token = 9
    bos_token = data_processing.Tokenizer().mapper["[CLS]"]
    reward_model = quark_finetune.RewardModel(
        DummyScorer(),
        eos_token=eos_token,
        prev_sequences=prev_sequences,
    )
    quark_model = quark_finetune.QuarkModel(
        train_model=DummyModel(),
        ref_model=DummyModel(),
        reward_model=reward_model,
        bos_token_ref=26,
    )
    quantiles = [[-100000, 0], [0, 0.25], [0.25, 0.5], [0.5, 1.5], [1.5, 100000]]
    quantile_offset = len(data_processing.Tokenizer().mapper)

    results = quark_finetune.generate_datapool_batch(
        quark_model,
        gen_kwargs={},
        eos_token=eos_token,
        quantile_offset=quantile_offset,
        quantiles=quantiles,
    )

    exp_results = [
        {
            "input_ids": [bos_token, 1, 2, 3, 4, 5, 6, eos_token, 0, 0],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            "quantile_token": 35,
        },
        {
            "input_ids": [bos_token, 1, 2, 3, 4, 5, 6, 7, 8, 10],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "quantile_token": 33,
        },
        {
            "input_ids": [bos_token, 5, 4, 3, 2, 1, eos_token, 0, 0, 0],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            "quantile_token": 32,
        }
    ]

    results = [
        {
            key: (value.cpu().tolist() if isinstance(value, torch.Tensor) else value) for key, value in r.items()
        } for r in results[0]
    ]
    assert(results == exp_results), f"{results} != {exp_results}"
    print("Test test_generate_datapool_batch passed")


def test_data_pool():
    data_pool_items = [
        {
            "input_ids": torch.LongTensor([26, 2, 3, 4, 5, 6, 7, 8, 9]),
            "attention_mask": torch.ByteTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            "quantile_token": 13,
        },
        {
            "input_ids": torch.LongTensor([26, 4, 1, 2, 3, 9, 0]),
            "attention_mask": torch.ByteTensor([1, 1, 1, 1, 1, 1, 0]),
            "quantile_token": 32,
        },
        {
            "input_ids": torch.LongTensor([26, 7, 9, 0]),
            "attention_mask": torch.ByteTensor([1, 1, 1, 0]),
            "quantile_token": 34,
        }
    ]
    pool = quark_finetune.DataPool(pool=data_pool_items, eos_token=9, max_length=6)

    exp_pool_items = [
        {
            "input_ids": torch.LongTensor([26, 4, 1, 2, 3, 9]),
            "attention_mask": torch.ByteTensor([1, 1, 1, 1, 1, 1]),
            "valid": True,
            "quantile_token": 32,
        },
        {
            "input_ids": torch.LongTensor([26, 7, 9, 0, 0, 0]),
            "attention_mask": torch.ByteTensor([1, 1, 1, 0, 0, 0]),
            "valid": True,
            "quantile_token": 34,
        }
    ]

    def convert_to_list(x: dict) -> dict:
        return {key: value.cpu().tolist() if isinstance(value, torch.Tensor) else value \
            for key, value in x.items()}

    obtained = [convert_to_list(p) for p in pool.pool]
    expected = [convert_to_list(p) for p in exp_pool_items]

    assert(obtained == expected), f"{obtained} != {expected}"

    train_sequences, ref_sequences = quark_finetune.collate_function([pool[0], pool[1]])

    exp_input_ids_collated_ref = torch.stack(
        [
            exp_pool_items[0]["input_ids"],
            exp_pool_items[1]["input_ids"]
        ]
    )

    exp_attention_mask_collated = torch.stack(
        [
            exp_pool_items[0]["attention_mask"],
            exp_pool_items[1]["attention_mask"]
        ]
    )

    assert(torch.all(ref_sequences["input_ids"] == exp_input_ids_collated_ref))
    assert(torch.all(ref_sequences["attention_mask"] == exp_attention_mask_collated))

    exp_input_ids_collated_train = exp_input_ids_collated_ref.clone()
    exp_input_ids_collated_train[0, 0] = 32
    exp_input_ids_collated_train[1, 0] = 34

    exp_train_items = {
        "input_ids": torch.LongTensor([
            [26, 32, 4, 1, 2, 3, 9],
            [26, 34, 7, 9, 0, 0, 0],
        ]),
        "attention_mask": torch.ByteTensor([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
        ])
    }

    def dict_eq(a, b):
        for key in a:
            assert(a[key].shape == b[key].shape), f"{key}: {a[key]} != {b[key]}"
            assert(torch.all(a[key] == b[key])), f"{key}: {a[key]} != {b[key]}"
        assert(len(a) == len(b))

    dict_eq(train_sequences, exp_train_items)

    print("Test test_data_pool passed")


def test_batched_membership():
    prev_sequences = torch.LongTensor(
        [
            [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 0, 0, 0],
            [1, 5, 4, 3, 1, 0],
        ]
    )

    test_sequences = torch.LongTensor(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 1, 1, 1, 1, 1],
            [1, 5, 4, 3, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    res = quark_finetune.membership_test_batched(prev_sequences, test_sequences, batch_size=2)

    assert(res.tolist() == [True, False, True, False])

    print("Test test_batched_membership passed")


def test_evaluation_loop():
    class DummyModel():
        def __init__(self, *args, **kwargs):
            self.call_num = 0

        def eval(self, *args, **kwargs):
            return self

        def generate(self, input_ids = None, bos_token_id = None, eos_token_id = None, gen_kwargs: dict = None):
            if bos_token_id is not None:
                bos = [bos_token_id]
            else:
                bos = input_ids.tolist()[0]

            if self.call_num == 0:
                res = torch.LongTensor(
                    [
                        bos + [1, 2, 3, 4, 5, 6, eos_token_id, -1, -2],
                        bos + [1, 2, 3, 4, 5, 6, 7, 8, 10],
                    ]
                )
            else:
                res = torch.LongTensor(
                    [
                        bos + [5, 4, 3, 2, 1, eos_token_id, 0, 0, 0],
                    ]
                )

            self.call_num += 1
            
            return res

    class DummyScorer:
        def __init__(self):
            self.call_num = 0

        def __call__(self, *args, **kwargs):
            if self.call_num == 0:
                res = torch.Tensor(
                    [[1.5], [0.5]],
                )
            else:
                res = torch.Tensor([[0.25]])

            self.call_num += 1
            return res

    prev_sequences = torch.ByteTensor(
        [
            [1, 2, 3, 4, 5, 6, 9],
            [3, 4, 5, 6, 9, 0, 0],
        ]
    )
    eos_token = 9
    reward_model = quark_finetune.RewardModel(
        DummyScorer(),
        eos_token=eos_token,
        prev_sequences=prev_sequences,
    )
    quark_model = quark_finetune.QuarkModel(
        train_model=DummyModel(),
        ref_model=DummyModel(),
        reward_model=reward_model,
        bos_token_ref=26,
    )
    quantiles = [[-100000, 0], [0, 0.25], [0.25, 0.5], [0.5, 1.5], [1.5, 100000]]
    quantile_offset = len(data_processing.Tokenizer().mapper)

    def do_nothing(array, *args, **kwargs):
        return array

    class DummyTrainer:
        def __init__(self):
            self.train_dataset = quark_finetune.DataPool(
                pool=[],
                eos_token=eos_token,
                fixed_epoch_length=True,
                max_length=8,
                pool_length=5,
                randsampler=do_nothing,
            )
            self.args = Namespace(
                n_eval_steps=2,
                quantiles=quantiles,
                gen_kwargs={},
                eos_token=eos_token,
                quantile_offset=quantile_offset,
                world_size=1,
                process_index=0,
            )
            self.model = quark_model

    trainer = DummyTrainer()

    assert(len(trainer.train_dataset) == 0)
    assert(len(trainer.train_dataset.pool) == 0)

    results = quark_finetune.evaluation_loop(
        trainer,
        dataloader=None,
        description=None,
        prediction_loss_only=False,
        ignore_keys=None,
        metric_key_prefix="eval",
    )

    assert(len(trainer.train_dataset.pool) == 2), f"{trainer.train_dataset.pool} is not of length 2 elements"

    exp_data_pool = [
        {
            "input_ids": [26, 1, 2, 3, 4, 5, 6, 9],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
            "quantile_token": 35,
            "valid": True,
        },
        {
            "input_ids": [26, 5, 4, 3, 2, 1, 9, 0],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0],
            "quantile_token": 32,
            "valid": True,
        }
    ]

    assert(to_list_helper(trainer.train_dataset.pool) == exp_data_pool), f"{trainer.train_dataset.pool} != {exp_data_pool}"

    exp_potentials = torch.Tensor([1.5, 0.5, 0.25])
    exp_membership = torch.Tensor([1, 0, 0])
    exp_valid = torch.Tensor([1, 0, 1])
    exp_prediction_results = torch.stack([exp_potentials, exp_membership, exp_valid], dim=1)
    assert(torch.all(results.predictions == exp_prediction_results)), f"{results.predictions} != {exp_prediction_results}"

    exp_all_valid_novel_potentials = 0.25 / 3
    exp_all_valid_potentials = 1.75 / 3
    exp_num_valid_novel = 1
    exp_num_valid = 2
    exp_metrics = {
        "eval_loss": [
            exp_all_valid_novel_potentials,
            exp_all_valid_potentials,
            exp_num_valid_novel,
            exp_num_valid
        ]
    }
    assert(results.metrics == exp_metrics), f"{results.metrics} != {exp_metrics}"

    assert(results.num_samples == 2)

    print("Test test_evaluation_loop passed")


def test_init_training():
    mapper = data_processing.Tokenizer().mapper
    eos_token = mapper["[SEP]"]

    class DummyModel():
        def __init__(self, *args, **kwargs):
            self.call_num = 0

        def eval(self, *args, **kwargs):
            return self

        def generate(self, input_ids = None, bos_token_id = None, eos_token_id = None, gen_kwargs: dict = None):
            if bos_token_id is not None:
                bos = [bos_token_id]
            else:
                bos = input_ids.tolist()[0]

            if self.call_num == 0:
                res = torch.LongTensor(
                    [
                        bos + [1, 2, 3, 4, 5, 6, eos_token_id, -1, -2],
                        bos + [1, 2, 3, 4, 5, 6, 7, 8, 10],
                    ]
                )
            else:
                res = torch.LongTensor(
                    [
                        bos + [5, 4, 3, 2, 1, eos_token_id, 0, 0, 0],
                    ]
                )

            self.call_num += 1
            
            return res

    class DummyScorer:
        def __init__(self):
            self.call_num = 0

        def __call__(self, *args, **kwargs):
            if self.call_num == 0:
                res = torch.Tensor(
                    [[1.5], [0.5]],
                )
            else:
                res = torch.Tensor([[0.25]])

            self.call_num += 1
            return res

    
    args = Namespace(
        n_eval_steps=2,
        quantiles=None,
        gen_kwargs={},
        eos_token=mapper["[SEP]"],
        quantile_offset=len(mapper),
        world_size=1,
        process_index=0,
    )

    prev_sequences = torch.ByteTensor(
        [
            [1, 2, 3, 4, 5, 6, eos_token],
            [3, 4, 5, 6, eos_token, 0, 0],
        ]
    )
    # eos_token = 9
    reward_model = quark_finetune.RewardModel(
        DummyScorer(),
        eos_token=eos_token,
        prev_sequences=prev_sequences,
    )
    quark_model = quark_finetune.QuarkModel(
        train_model=DummyModel(),
        ref_model=DummyModel(),
        reward_model=reward_model,
        bos_token_ref=26,
    )

    results = quark_finetune.init_training(
        quark_model,
        quantile_spec=[0.33, 0.66, 1.00],
        n_batches_init=2,
        gen_kwargs={},
        pool_size=5,
        max_length=14,
        args=args,
    )

    exp_data_pool = [
        {
            "input_ids": [26, 1, 2, 3, 4, 5, 6, eos_token, 0, 0, 0, 0, 0, 0],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "quantile_token": 34,
            "valid": True,
        },
        {
            "input_ids": [26, 5, 4, 3, 2, 1, eos_token, 0, 0, 0, 0, 0, 0, 0],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "quantile_token": 31,
            "valid": True,
        }
    ]

    exp_quantile = list(reversed([
        (1.5, float("inf")),  # The top sequence is 1.5. This is selected to include in this bucket
        (0.5, 1.5),  # The second sequence is 0.5. This is selected to include here. Upper bound is lower-bound of previous
        (0.25, 0.5),  # The third sequence is this one, and same logic as above for the selection
        (-float("inf"), 0.25),  # Terminal quantile to catch anything that falls outside the initial range
    ]))

    assert(results[1] == exp_quantile), f"{results[1]} != {exp_quantile}"

    assert(to_list_helper(results[0].pool) == exp_data_pool), f"{results[0].pool} != {exp_data_pool}"

    print("Test test_init_training passed")


def test_tensorize_prior_sequences():
    testdir = "/tmp/test_tensorize_prior_sequences"
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    os.makedirs(testdir)

    with open(os.path.join(testdir, "fasta.fa"), "w") as fhandle:
        testfile = fhandle.name
        fhandle.write(">1\nACFGA*\n>2\nBCF*")

    res = quark_finetune.tensorize_prior_sequences(testfile, max_length=8)

    mapper = data_processing.Tokenizer().mapper

    exp_results = torch.LongTensor([
        [mapper[x] for x in "ACFGA*"],
        [mapper[x] for x in "BCF*"] + [0, 0],
    ])

    assert(torch.all(exp_results == res))

    shutil.rmtree(testdir)

    print("Test test_tensorize_prior_sequences passed")


def test_pregen_to_pool():
    testdir = "/tmp/test_pregen_to_pool"

    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    os.makedirs(testdir)

    with open(os.path.join(testdir, "pregen.json"), "w") as fhandle:
        fhandle.write(
            json.dumps({"seq": "ACGHACTHACH*", "ll": -0.10}) + "\n"
        )
        fhandle.write(
            json.dumps({"seq": "XAFHACT*", "ll": -0.90}) + "\n"
        )
        fhandle.write(
            json.dumps({"seq": "XBXXCT*", "ll": -1.90})
        )
        pregen_file = fhandle.name

    class DummyModel:
        def __init__(self):
            self.is_cuda = False


    res_gen = quark_finetune.convert_pregen_to_pool(
        pregen_file,
        DummyModel(),
        max_length=9,
        batch_size=2,
        training_args=Namespace(
            world_size=2,
            process_index=0,
        )
    )

    n_batches = next(res_gen)
    assert(n_batches == 1)

    batch = next(res_gen)

    try:
        next(res_gen)
        raise ValueError("StopIteration not triggered")
    except StopIteration:
        pass

    mapper = data_processing.Tokenizer().mapper

    exp = torch.LongTensor([
        [mapper["[CLS]"]] + [mapper[i] for i in "ACGHACTH"],
        [mapper["[CLS]"]] + [mapper[i] for i in "XBXXCT*"] + [0]
    ])

    assert(torch.all(exp == batch))

    print("Test test_pregen_to_pool passed")


if __name__ == "__main__":
    test_pad_equally()
    test_reward_model()
    test_get_attn_mask()
    test_quark_forward()
    test_generate_datapool_batch()
    test_data_pool()
    test_evaluation_loop()
    test_init_training()
    test_tensorize_prior_sequences()
    test_pregen_to_pool()
    test_batched_membership()
