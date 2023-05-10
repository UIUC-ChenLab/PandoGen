# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    PretrainedConfig,
    EncoderDecoderModel,
    __version__,
)
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field, asdict
from data_processing import Dataset, collate_function, Tokenizer, CompoundDataset
import data_processing
from typing import Tuple, Optional, List
import models
import logging
import os
import pickle
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
import torch
from transformers.trainer_pt_utils import get_parameter_names
from collections import defaultdict
from argparse import Namespace
import re
import random

logger = logging.getLogger(name=__file__)


def filter_size_mismatches(trainer: Trainer, state_dict: dict, tgt_state_dict: Optional[dict] = None) -> None:
    if tgt_state_dict is None:
        tgt_state_dict = trainer.model.state_dict()

    remove_keys = []

    logger.info(f"Total number of keys in original load point = {len(state_dict)}")

    for key in state_dict:
        # Remove keys that aren't there in the model
        if key not in tgt_state_dict:
            logger.info(f"Removing key {key} due to not being present in target")
            remove_keys.append(key)
        # Remove keys with size mismatch
        elif state_dict[key].shape != tgt_state_dict[key].shape:
            logger.info(f"Removing key {key} due to size mismatch")
            remove_keys.append(key)

    for key in remove_keys:
        del state_dict[key]

    logger.info(f"Total number of keys in compatible load point = {len(state_dict)}")


def load_from_checkpoint(trainer: Trainer, checkpoint: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
    """
    Logic copied from https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L1125
    """
    if not state_dict:
        if checkpoint is None:
            return

        if not os.path.isfile(os.path.join(checkpoint, WEIGHTS_NAME)):
            raise ValueError(f"Can't find a valid checkpoint at {checkpoint}")

        logger.info(
            f"Loading model from checkpoint={checkpoint})."
            f"checkpoint={checkpoint} is overridden by training_args.resume_from_checkpoint,"
            f"and any checkpoints in the output directory."
        )

        if os.path.isfile(os.path.join(checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warn(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME), map_location="cpu")

    try:
        trainer._load_state_dict_in_model(state_dict)
    except RuntimeError:
        logger.warning(
            "Some weights in checkpoint do not match model sizes. Doing partial load.")
        filter_size_mismatches(trainer, state_dict)
        res = trainer.model.load_state_dict(state_dict, strict=False)
        logger.warning(f"Following keys weren't loaded {res}")


def get_layerwise_groupings(model: EncoderDecoderModel) -> dict:
    """
    Output structure:

    non_attn_parameters: {
        decay_parameters: []
        nondecay_parameters: []
    },
    attn_parameters: {
        encoder_layers: {}
            0: {decay_parameters: [], nondecay_parameters: []},  # layer0 parameters
            1: {decay_parameters: [], nondecay_parameters: []},  # layer1 parameters
            ...
        },
        decoder_layers: {
            0: {decay_parameters: [], nondecay_parameters: []},  # layer0 parameters
            1: {decay_parameters: [], nondecay_parameters: []},  # layer1 parameters
            ...
        }
    }
    """
    grouped_params = {
        "non_attn_parameters": {"decay_parameters": [], "nondecay_parameters": []},
        "attn_parameters": {
            "encoder_layers": defaultdict(lambda: {"decay_parameters": [], "nondecay_parameters": []}),
            "decoder_layers": defaultdict(lambda: {"decay_parameters": [], "nondecay_parameters": []}),
        }
    }

    decay_parameters = [param_name for param_name in get_parameter_names(
        model, [torch.nn.LayerNorm]) if "bias" not in param_name]
    
    for param_name, parameter in model.named_parameters():
        param_decay_type = "decay_parameters" if param_name in decay_parameters \
            else "nondecay_parameters"

        """
        If the layer is clearly an encoder or decoder layer, we know its
        layer number from regexp. If it is an embedding layer, and if it is
        not a positional embedding parameter, we stack it with layer 0 for
        encoder or decoder. If it is positional_embeddings, we club it with
        the non_attn case by setting layer_num to None.
        """
        layer_num = re.findall(r"encoder\.layer\.(\d+)\.", param_name)

        if layer_num:
            layer_num = int(layer_num.pop())
        elif ".embeddings." in param_name:
            if ".position_embeddings." in param_name:
                layer_num = None
            else:
                layer_num = 0
        else:
            layer_num = None 

        if type(layer_num) is int:
            if param_name.startswith("encoder"):
                grouped_params["attn_parameters"]["encoder_layers"][layer_num][param_decay_type].append(parameter)
            elif param_name.startswith("decoder"):
                grouped_params["attn_parameters"]["decoder_layers"][layer_num][param_decay_type].append(parameter)
            else:
                grouped_params["non_attn_parameters"][param_decay_type].append(parameter)
        else:
            grouped_params["non_attn_parameters"][param_decay_type].append(parameter)

    return grouped_params


def get_finetuning_optimizer(
    model: EncoderDecoderModel,
    lr: float,
    lr_decay_rate: float,
    weight_decay: float,
    optimizer_kwargs: dict,
) -> torch.optim.AdamW:
    """
    Create decaying learning rates for top to bottom layers

    Notes:
    PyTorch code for AdamW: https://pytorch.org/docs/1.10/_modules/torch/optim/adamw.html#AdamW
        This indicates the necessary items to include in parameter groups

    Transformers code for linear scheduler: https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/optimization.py#L75
        This shows how linear scheduler is created, for example. This is simply a Lambda scheduler. 
        Also note that the lambda function is separately called for each parameter group in the optimizer.
        Hence we only need to create different parameter groups with different learning rates. The file `test_lr_schedule.py`
        shows that lr is properly initialized for each parameter group for warmup cases.

    Transformers code for creating optimizer: https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/trainer.py#L806
        LayerNorm and bias parameters are excluded from decay. We will do the same here.
    """
    grouped_parameters = get_layerwise_groupings(model)

    parameter_groups = []

    parameter_groups.extend([
        {"params": grouped_parameters["non_attn_parameters"]["decay_parameters"], "lr": lr, "weight_decay": weight_decay},
        {"params": grouped_parameters["non_attn_parameters"]["nondecay_parameters"], "lr": lr},
    ])

    for submodule_type in ["encoder_layers", "decoder_layers"]:
        current_lr = lr

        for key in sorted(grouped_parameters["attn_parameters"][submodule_type].keys(), reverse=True):
            group = grouped_parameters["attn_parameters"][submodule_type][key]
            parameter_groups.extend([
                {"params": group["decay_parameters"], "lr": current_lr, "weight_decay": weight_decay},
                {"params": group["nondecay_parameters"], "lr": current_lr},
            ])
            current_lr = current_lr * lr_decay_rate

    return torch.optim.AdamW(parameter_groups, **optimizer_kwargs)


@dataclass
class ModelArguments:
    hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "Hidden size (or embedding size) of the model"}
    )

    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={"help": "Number of hidden layers in encoder"}
    )
    
    num_hidden_layers_decoder: Optional[int] = field(
        default=None,
        metadata={"help": "Number of decoder hidden layers. Defaults to num_hidden_layers"}
    )

    num_attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "Number of attention heads"}
    )

    intermediate_size: Optional[int] = field(
        default=3072,
        metadata={"help": "Intermediate FC layer size"}
    )

    hidden_act: Optional[str] = field(
        default="gelu",
        metadata={"help": "Activation"}
    )

    hidden_dropout_prob: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout for hidden layers"}
    )

    attention_probs_dropout_prob: Optional[float] = field(
        default=0.1,
        metadata={"help": "Attention probability dropout rate"}
    )

    max_position_embeddings: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of positional embeddings to use"}
    )

    checkpoint_params: Optional[str] = field(
        default=None,
        metadata={"help": "Load parameters of model from checkpoint"}
    )

    checkpoint_model_type: Optional[str] = field(
        default="EncoderDecoder",
        metadata={"help": "EncoderDecoder/Decoder model type for loading"}
    )

    lr_decay_rate: Optional[float] = field(
        default=1.0,
        metadata={"help": "Decay learning rate from the top layer down"}
    )

    num_quark_quantiles: Optional[int] = field(
        default=None,
        metadata={"help": "Number of quantiles for quark fine-tuning"}
    )


@dataclass
class DataArguments:
    train: Optional[str] = field(
        default=None,
        metadata={"help": "Path to training data directory"}
    )

    val: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation data directory"}
    )

    test: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test data directory"}
    )

    distributed_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Distributed dataset dump path"}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of training examples"}
    )

    max_val_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of validation examples"}
    )

    def __post_init__(self):
        if not(self.train or self.val or self.test or self.distributed_dataset):
            raise ValueError("Set one data argument")
        if (self.train or self.val or self.test) and (self.distributed_dataset):
            raise ValueError("Do not set regular train/val/split and distributed_dataset")


def make_data(data_args: DataArguments) -> tuple:
    if data_args.train:
        logger.info("Regular dataset provided. Loading.")
        train_data = Dataset(data_args.train)
        val_data = Dataset(data_args.val)
        test_data = Dataset(data_args.test)
    else:
        logger.info("Distributed dataset provided. Loading.")
        train_data = CompoundDataset(data_args.distributed_dataset, "train")
        val_data = CompoundDataset(data_args.distributed_dataset, "val")
        test_data = CompoundDataset(data_args.distributed_dataset, "test")

    return (train_data, val_data, test_data)


def make_trainer(model: torch.nn.Module, data: tuple, training_args: TrainingArguments, model_args: ModelArguments):
    optimizers = (None, None)

    if 0 <= model_args.lr_decay_rate < 1:
        logger.info("Creating layer-wise parameter groups")
        optimizer_kwargs = {
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
        }

        optimizers = (
            get_finetuning_optimizer(
                model,
                lr=training_args.learning_rate,
                lr_decay_rate=model_args.lr_decay_rate,
                weight_decay=training_args.weight_decay,
                optimizer_kwargs=optimizer_kwargs,
            ),
            None
        )

    return Trainer(
        model,
        args=training_args,
        data_collator=collate_function,
        train_dataset=data[0],
        eval_dataset=data[1],
        optimizers=optimizers,
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    random.seed(training_args.seed)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Checkpoint resumption logic from run_clm.py sample script
    # https://github.com/huggingface/transformers/blob/v4.16.2-release/examples/pytorch/language-modeling/run_clm.py
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logger.info("Initializing data, models and trainer")
    data = make_data(data_args)
    train_dataset, val_dataset, test_dataset = data
    orig_tokenizer = train_dataset.data.tokenizer
    model = models.create_encoder_decoder_model(
        train_dataset.data.tokenizer, **asdict(model_args))

    if data_args.max_train_samples:
        # train_dataset.set_length(data_args.max_train_samples)
        logger.info(f"Setting training dataset size to {data_args.max_train_samples}")
        train_dataset = torch.utils.data.Subset(
            train_dataset,
            indices=random.sample(range(len(train_dataset)), k=data_args.max_train_samples),
        )

    if data_args.max_val_samples:
        # val_dataset.set_length(data_args.max_val_samples)
        logger.info(f"Setting validation dataset size to {data_args.max_val_samples}")
        val_dataset = torch.utils.data.Subset(
            val_dataset,
            indices=random.sample(range(len(val_dataset)), k=data_args.max_val_samples),
        )

    data = (train_dataset, val_dataset, test_dataset)
    trainer = make_trainer(model, data, training_args, model_args)
    num_params = sum(x.numel() for x in model.parameters())
    logger.info(f"Initialized model with {num_params} parameters")
    load_from_checkpoint(trainer, model_args.checkpoint_params)

    if training_args.do_train:
        ### Checkpoint resumption logic from run_clm.py
        # https://github.com/huggingface/transformers/blob/v4.16.2-release/examples/pytorch/language-modeling/run_clm.py
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Save tokenizer separately as it is not an AutoTokenizer.
        # Also tokenizer is not necessary during training or evaluation
        # It is only needed for generation from fresh sequences. It is also
        # available in the data directory, but we save it here for convenience.
        with open(os.path.join(training_args.output_dir, "tokenizer.pkl"), "wb") as fhandle:
            pickle.dump(orig_tokenizer, fhandle)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["val_samples"] = len(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s")
    main()
