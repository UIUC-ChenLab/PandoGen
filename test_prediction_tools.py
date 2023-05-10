# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import prediction_tools


def test_prepare_stimuli():
    seq = "ACGBHTACT"
    window_size = 3
    step_size = 2
    frozen_locations = [4, 5]

    results = [
        ["[CLS],[MASK],B,H,T,A,C,T,[SEP]", "A,C,G,[MASK]"],
        ["[CLS],A,C,[MASK],H,T,A,C,T,[SEP]", "G,B,[MASK]"],
        ["[CLS],A,C,G,B,H,T,[MASK],C,T,[SEP]", "A,[MASK]"],
        ["[CLS],A,C,G,B,H,T,[MASK],T,[SEP]", "A,C,[MASK]"],
    ]

    res = prediction_tools.prepare_stimulii(
        seq, window_size, step_size, frozen_locations=frozen_locations
    )

    assert(len(res) == len(results))

    for r1, r2 in zip(res, results):
        assert(r1.encoder_seq == r2[0].split(",")), f"{r1.encoder_seq}, {r2[0]}"
        assert(r1.decoder_seq == r2[1].split(",")), f"{r1.encoder_seq}, {r2[0]}"

    print("Test test_prepare_stimulii passed")


def test_prepare_stimuli_gapped():
    seq = "ACGBHTACT"
    window_size = 3
    step_size = 2
    frozen_locations = [4, 6]

    results = [
        ["[CLS],[MASK],B,H,T,A,C,T,[SEP]", "A,C,G,[MASK]"],
        ["[CLS],A,C,[MASK],H,T,A,C,T,[SEP]", "G,B,[MASK]"],
        ["[CLS],A,C,G,B,H,[MASK],A,C,T,[SEP]", "T,[MASK]"],
        ["[CLS],A,C,G,B,H,T,A,[MASK],T,[SEP]", "C,[MASK]"],
    ]

    res = prediction_tools.prepare_stimulii(
        seq, window_size, step_size, frozen_locations=frozen_locations
    )

    assert(len(res) == len(results))

    for r1, r2 in zip(res, results):
        assert(r1.encoder_seq == r2[0].split(",")), f"{r1.encoder_seq}, {r2[0]}"
        assert(r1.decoder_seq == r2[1].split(",")), f"{r1.encoder_seq}, {r2[0]}"

    print("Test test_prepare_stimulii_gapped passed")


def test_calc_likelihoods():
    import torch
    decoded = torch.LongTensor(
        [
            [0, 1, 2, 1],
            [1, 2, 1, 0],
            [0, 1, 0, 1],
        ]
    )

    def get_mode(i: int) -> list:
        return [1e-1 / 2 if k != i else 1 - 1e-1 for k in range(3)]

    probs = torch.Tensor(
        [
            [get_mode(0), get_mode(1), get_mode(2), get_mode(0)],
            [get_mode(1), get_mode(2), get_mode(1), get_mode(1)],
            [get_mode(0), get_mode(1), get_mode(0), get_mode(1)],
        ]
    )
    # Create logits
    logits = torch.log(probs / (1 - probs))
    # Recreate logits (to avoid floating point issues)
    probs = torch.softmax(logits, dim=2)

    res = prediction_tools.calc_likelihoods(decoded, end_token_value=2, logits=logits)
    exp = [
        torch.log(probs[0, 0, 0] * probs[0, 1, 1] * probs[0, 2, 2]),
        torch.log(probs[1, 0, 1] * probs[1, 1, 2]),
        -1e5
    ]

    for i, j in zip(res.tolist(), exp):
        assert(i - 1e-4 <= j <= i + 1e-4)

    print("Test test_calc_likelihoods passed")


def test_predict():
    import data_processing
    import torch
    from argparse import Namespace

    tokenizer = data_processing.Tokenizer()

    class Model:
        def __init__(self):
            self.logits = torch.Tensor([list(range(len(tokenizer.mapper))) for i in range(6)])

        def generate(self, *args, **kwargs):
            return torch.LongTensor(
                [[tokenizer.mapper[i] for i in ["[CLS]", "G", "G", "[MASK]", "C", "G"]]])

        def parameters(self):
            return iter([torch.Tensor([1])])

        def __call__(self, *args, **kwargs):
            """
            See comment in batch calculate to see how it goes
            """
            if (kwargs['decoder_input_ids'] is not None) and (kwargs['labels'] is None):
                logits = self.logits[None]
                return Namespace(logits=logits)
            elif (kwargs['decoder_input_ids'] is None) and (kwargs['labels'] is not None):
                logits = self.logits[None, :4]
                return Namespace(logits=logits)
            else:
                raise AttributeError(
                    "Both decoder_input_ids and labels should not be not None or None")

    seq = "ACGBHTACT"

    stimulus = prediction_tools.Stimulus(
        seq=seq,
        mask_position=0,
        mask_length=3,
        encoder_seq="[CLS],[MASK],B,H,T,A,C,T,[SEP]".split(","),
        decoder_seq="A,C,G,[MASK]".split(","),
    )

    forward_mapper = tokenizer.mapper
    reverse_mapper = {}

    for k, v in forward_mapper.items():
        if v not in reverse_mapper:
            reverse_mapper[v] = k

    model = Model()

    results = prediction_tools.predict(
        model,
        [stimulus],
        forward_mapper=forward_mapper,
        reverse_mapper=reverse_mapper,
    )

    a_index = tokenizer.mapper["A"]
    c_index = tokenizer.mapper["C"]
    g_index = tokenizer.mapper["G"]
    mask_index = tokenizer.mapper["[MASK]"]
    logits = model.logits[:4]
    dist = torch.distributions.categorical.Categorical(logits=logits)
    res = dist.log_prob(torch.LongTensor([a_index, c_index, g_index, mask_index]))
    expected_ll = res[2] + res[2] + res[-1]
    expected_ll_decoder = res[0] + res[1] + res[2] + res[-1]
    expected_ll_ratio = expected_ll - expected_ll_decoder
    
    assert(len(results) == 1)
    assert(results[0].seq == seq)
    assert(results[0].mask_position == 0)
    assert(results[0].mask_length == 3)
    assert(results[0].encoder_seq == stimulus.encoder_seq)
    assert(results[0].decoder_seq == stimulus.decoder_seq)
    assert(results[0].predicted_sequences == [["G", "G"]])
    assert(results[0].differing_sequences == [["G", "G"]])
    assert(results[0].get_mutation_seq(seq) == ["A1del,C2G"])

    def shape_1_tester(x):
        assert(len(x.shape) == 1)
        assert(x.shape[0] == 1)

    shape_1_tester(results[0].predicted_lls)
    shape_1_tester(results[0].predicted_ll_ratios)
    shape_1_tester(results[0].differing_lls)
    shape_1_tester(results[0].differing_ll_ratios)

    def float_eq_helper(a, b, eps=1e-4):
        assert(a - eps <= b <= a + eps)

    float_eq_helper(results[0].predicted_ll_ratios.item(), expected_ll_ratio.item())
    float_eq_helper(results[0].predicted_lls.item(), expected_ll.item())
    float_eq_helper(results[0].differing_ll_ratios.item(), expected_ll_ratio.item())
    float_eq_helper(results[0].differing_lls.item(), expected_ll.item())

    print("Test test_predict passed")
    

if __name__ == "__main__":
    test_prepare_stimuli()
    test_prepare_stimuli_gapped()
    test_calc_likelihoods()
    test_predict()
