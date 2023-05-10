# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertModel, BertLMHeadModel
from utils import SpecialTokens, _DEFAULT_SPECIAL_TOKENS, AMINO_ACIDS
import copy
from data_processing import Tokenizer
from typing import Optional, Union
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
import os
import torch

"""
NOTE: We are using opence-v1.6.1 which has transformers version 4.16.2, pytorch 1.10.2

Code References
---------------

Basic instructions on creating encoder-decoder models:
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/encoder-decoder
    (also states instantiation from config does not cause initialization to pretrained values)

BERT model source comment blurb clarifying use as encoder and decoder:
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/bert/modeling_bert.py#L848

EncoderDecoderModel left-zero-pads tokens to produce decoder inputs:
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L508

What happens in EncoderDecoderConfig:
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/encoder_decoder/configuration_encoder_decoder.py#L92

CrossEntropyLoss is used to calculate decoder loss. CrossEntropyLoss ignores value -100
    https://pytorch.org/docs/1.10/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss

Trainer uses DistributedSampler to split dataset for multiprocess training
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L623

For the above, process_index is used. But torch.distributed.launch sets local_rank. Here is the conversion
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/training_args.py#L1137
"""


def create_encoder_decoder_model(
    tokenizer: Tokenizer,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    hidden_act: str = "gelu",
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    max_position_embeddings: int = 1024,
    num_hidden_layers_decoder: Optional[int] = None,
    **kwargs,
) -> EncoderDecoderModel:
    if not tokenizer:
        tokenizer = Tokenizer()
    base_config = BertConfig(
        vocab_size=len(AMINO_ACIDS) + tokenizer.special_tokens.num_special_tokens,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
    )
    decoder_config = copy.deepcopy(base_config)
    if num_hidden_layers_decoder:
        decoder_config.num_hidden_layers = num_hidden_layers_decoder
    config = EncoderDecoderConfig.from_encoder_decoder_configs(base_config, decoder_config)
    model = EncoderDecoderModel(config=config)
    model.config.pad_token_id = tokenizer.mapper[tokenizer.special_tokens.end_of_sequence]
    model.config.decoder_start_token_id = tokenizer.mapper[
        tokenizer.special_tokens.start_of_sequence]
    return model


def create_decoder_model(
    tokenizer: Tokenizer,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    hidden_act: str = "gelu",
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    max_position_embeddings: int = 1024,
    num_quark_quantiles: Optional[int] = None,
    **kwargs,
) -> BertConfig:
    if not tokenizer:
        tokenizer = Tokenizer()

    n = 0 if not num_quark_quantiles else num_quark_quantiles

    decoder_config = BertConfig(
        vocab_size=len(AMINO_ACIDS) + tokenizer.special_tokens.num_special_tokens + n,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        is_decoder=True,
        add_cross_attention=False,
    )

    model = BertLMHeadModel(decoder_config)

    return model


def from_pretrained(checkpoint: str, model_type: str = "EncoderDecoder") -> Union[EncoderDecoderModel, BertModel]:
    """
    Load a pre-trained EncoderDecoderModel
    """
    if model_type == "EncoderDecoder":
        cfg_class = EncoderDecoderConfig
        model_class = EncoderDecoderModel
    else:
        cfg_class = BertConfig
        model_class = BertLMHeadModel

    cfg = cfg_class.from_json_file(os.path.join(checkpoint, CONFIG_NAME))
    model = model_class(cfg)
    state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME), map_location="cpu")
    model.load_state_dict(state_dict)
    return model
