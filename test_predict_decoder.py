# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import predict_decoder
import torch
from argparse import Namespace
import quark_finetune


def test_quark_predictor():
    class DummyModel:
        def __init__(self):
            pass

        def generate(
            self,
            input_ids = None,
            bos_token_id = None,
            eos_token_id = None,
            **kwargs
        ):
            if input_ids is not None:
                input_ids_list = input_ids.cpu().tolist()[0]
                return torch.LongTensor(
                    [
                        input_ids_list + [1, 2, 3, 4, eos_token_id]
                    ]
                )
            else:
                return torch.LongTensor(
                    [
                        [bos_token_id, 1, 2, 3, 4, eos_token_id]
                    ]
                )

        def forward(self, *args, **kwargs):
            return None

    class QuarkModel:
        def __init__(self):
            self.train_model = DummyModel()


    predictor = predict_decoder.QuarkPredictor(QuarkModel(), quantile_token=34)
    res = predictor.generate(bos_token_id=26, eos_token_id=27, gen_kwargs={})
    assert(res.cpu().tolist()[0] == [26, 34, 1, 2, 3, 4, 27])

    print("Test test_quark_predictor passed")


def test_calc_likelihoods_top():
    # predict_decoder._EOS_TOKEN = 2

    seq0 = [26, 1, 0, 0, 1, 2, 0]
    seq1 = [26, 1, 0, 1, 0, 1, 2]
    logits0 = [
        [0, 1, 2],
        [1, -1, 0],
        [0, 1, 2],
        [1, 2, 3],
        [-1, 2, 1],
        [0, -1, 1],
        [-1, -1, -1],
    ]

    logits1 = [
        [0, 1, 2],
        [1, 0, -1],
        [0, 2, 1],
        [3, 1, 2],
        [1, -1, 2],
        [-1, 1, 0],
        [-1, -2, -3]
    ]

    seq = torch.LongTensor([seq0, seq1])
    logits = torch.Tensor([logits0, logits1])

    class DummyModel:
        def __init__(self):
            pass

        def __call__(self, *args, **kwargs):
            return Namespace(logits=logits)

    def get_ll(logits, idx):
        return torch.log_softmax(torch.Tensor(logits), dim=0)[idx].item()

    exp_ll_quark = [0, 0]

    for i in range(5):
        if i < 4:
            exp_ll_quark[0] += get_ll(logits0[i + 1], seq0[i + 2])
        exp_ll_quark[1] += get_ll(logits1[i + 1], seq1[i + 2])

    ll_quark = quark_finetune.calc_likelihoods_top(
        DummyModel(),
        seq,
        is_quark_model=True,
        eos_token=2,
    )

    def float_eq(a, b, eps=1e-4):
        return a - eps <= b <= a + eps

    assert(float_eq(ll_quark[0].item(), exp_ll_quark[0])), f"{ll_quark[0]} != {exp_ll_quark[0]}"
    assert(float_eq(ll_quark[1].item(), exp_ll_quark[1])), f"{ll_quark[1]} != {exp_ll_quark[1]}"

    exp_ll_non_quark = [0, 0]

    for i in range(6):
        if i < 5:
            exp_ll_non_quark[0] += get_ll(logits0[i], seq0[i + 1])
        exp_ll_non_quark[1] += get_ll(logits1[i], seq1[i + 1])

    ll_non_quark = quark_finetune.calc_likelihoods_top(
        DummyModel(),
        seq,
        is_quark_model=False,
        eos_token=2,
    )

    assert(float_eq(ll_non_quark[0].item(), exp_ll_non_quark[0])), f"{ll_non_quark[0]} != {exp_ll_non_quark[0]}"
    assert(float_eq(ll_non_quark[1].item(), exp_ll_non_quark[1])), f"{ll_non_quark[1]} != {exp_ll_non_quark[1]}"

    print("Test test_calc_likelihoods_top passed")


if __name__ == "__main__":
    test_quark_predictor()
    test_calc_likelihoods_top()
