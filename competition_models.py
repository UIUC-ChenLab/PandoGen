# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import torch
from transformers import Trainer, EncoderDecoderModel, BertLMHeadModel, BertModel
import models
import logging
from typing import Optional, Union

logger = logging.getLogger(__file__)


def kl_divergence(target: torch.Tensor, prediction_proba: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    KL-Divergence = sum_x P(x) [log P(x) - log Q(x)]
    Here, we set P(x) to be the target, and Q(x) to be the prediction. We want to minimize
    the KL-Divergence between the targer distribution and the predicted distribution.
    """
    return torch.mean(
        torch.sum(target * (
            torch.log(target + eps) - 
            torch.log(prediction_proba + eps)
        ), dim=1)
    )


class SimplePooler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.ByteTensor, eps: float = 1e-12) -> torch.Tensor:
        """
        :param hidden_state: Hidden state of shape [batch_size, length, #dim]
        :param attention_mask: Hidden state of shape [batch_size, length]
        """
        hidden_state_masked = hidden_state.masked_fill(attention_mask[:, :, None] == 0, 0)
        return torch.sum(hidden_state_masked, dim=1) / (
            torch.sum(attention_mask, dim=1, keepdim=True) + eps)


def get_last_item(seq: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
    mask_shifted = torch.roll(mask, shifts=-1, dims=1)
    mask_shifted[:, -1] = 0
    last_position_mask = mask - mask_shifted
    return seq.masked_select(last_position_mask[:, :, None] == 1).view(seq.shape[0], seq.shape[-1])


class DecoderPooler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.ByteTensor, *args, **kwargs):
        return get_last_item(hidden_state, attention_mask)


class ClsPooler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_state: torch.Tensor, *args, **kwargs):
        return hidden_state[:, 0]


class CompetitionModel(torch.nn.Module):
    def __init__(
        self,
        core_model: BertModel,
        pooler: torch.nn.Module,
        referee: torch.nn.Module,
        prediction_type: str = "weight",
    ):
        super().__init__()
        self.core_model = core_model
        self.pooler = pooler
        self.referee = referee
        self.prediction_type = prediction_type

    @property
    def prediction_type(self) -> str:
        return self._prediction_type

    @prediction_type.setter
    def prediction_type(self, _prediction_type: str):
        if _prediction_type not in ["binary", "weight"]:
            raise AttributeError(f"prediction_type should be either binary, or weight. Got {_prediction_type}")
        self._prediction_type = _prediction_type

    def forward(
        self,
        seq_a_tokens: dict,
        seq_b_tokens: dict,
        seq_a_leading_weeks: Optional[list] = None,
        seq_b_leading_weeks: Optional[list] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        res_a = self.core_model(**seq_a_tokens, output_hidden_states=True)
        res_b = self.core_model(**seq_b_tokens, output_hidden_states=True)
        pooled_a = self.pooler(res_a.last_hidden_state, seq_a_tokens["attention_mask"])
        pooled_b = self.pooler(res_b.last_hidden_state, seq_b_tokens["attention_mask"])

        if (seq_a_leading_weeks is None and seq_b_leading_weeks is not None) or (
            seq_b_leading_weeks is None and seq_a_leading_weeks is not None):
            raise AttributeError("One of seq_a_leading_weeks, seq_b_leading_weeks is None")

        if seq_a_leading_weeks is not None:
            pooled_a = torch.cat([pooled_a, seq_a_leading_weeks], dim=1)
            pooled_b = torch.cat([pooled_b, seq_b_leading_weeks], dim=1)

        return {
            "logits": self.referee(pooled_a, pooled_b)}


class CompetitionTrainer(Trainer):
    def compute_loss(self, model: CompetitionModel, inputs: dict, return_outputs: bool = False) -> torch.Tensor:
        try:
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            if logits.shape[-1] > 1:
                prob = torch.softmax(logits, dim=1)
            else:
                prob1 = torch.sigmoid(logits)
                prob0 = 1 - prob1
                prob = torch.cat([prob0, prob1], dim=1)

            loss_fn = torch.nn.CrossEntropyLoss()

            if self.model.prediction_type == "binary":
                labels = (labels > 0.5).long()
            else:
                labels = torch.stack([1 - labels, labels], dim=1)

            loss = loss_fn(prob, labels)

        except Exception as e:
            logger.error(f"Failed to compute for inputs: {inputs}")
            raise e

        if return_outputs:
            return loss, outputs
        else:
            return loss


class Predictor(torch.nn.Module):
    """
    Model for competition prediction
    """
    def __init__(self, model: CompetitionModel):
        super().__init__()
        self.referee = model.referee

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        return self.referee(embeddings_a, embeddings_b)


class Embedder(torch.nn.Module):
    """
    Model for competition prediction
    """
    def __init__(self, model: CompetitionModel):
        super().__init__()
        self.core_model = model.core_model
        self.pooler = model.pooler

    def forward(self, sequence_tokens: dict, *args, **kwargs) -> torch.Tensor:
        res = self.core_model(**sequence_tokens, output_hidden_states=True)
        pooled = self.pooler(res.last_hidden_state, sequence_tokens["attention_mask"])
        return pooled


class PredictPotential(torch.nn.Module):
    """
    Model for predicting potentials for a predicted sequence
    """
    def __init__(self, model: CompetitionModel):
        super().__init__()
        self.mixer = model.referee.mixer
        self.linear = model.referee.linear

    def forward(self, embeddings: torch.Tensor, leading_weeks: Optional[torch.LongTensor] = None, **kwargs) -> torch.Tensor:
        try:
            if leading_weeks is not None:
                inputs = torch.cat((embeddings, leading_weeks), dim=1)
            else:
                inputs = embeddings

            res = self.linear(self.mixer(inputs))
        except Exception as e:
            logger.error("Referee is not binary!")
            raise e

        return res


class Scorer(torch.nn.Module):
    """
    Single scorer module
    """
    def __init__(self, model: CompetitionModel):
        super().__init__()
        self.embedder = Embedder(model)
        self.potential_predictor = PredictPotential(model)

    def forward(self, sequence_tokens: dict, *args, **kwargs) -> torch.Tensor:
        embeddings = self.embedder(sequence_tokens)
        return self.potential_predictor(embeddings)


class SimpleReferee(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(2 * hidden_size, 2)

    def forward(self, pooled_a: torch.Tensor, pooled_b: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([pooled_a, pooled_b], dim=1))


class TripletReferee(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(3 * hidden_size, 2)

    def forward(self, pooled_a: torch.Tensor, pooled_b: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([pooled_a, pooled_b, pooled_a - pooled_b], dim=1))


class BinaryReferee(torch.nn.Module):
    def __init__(self, hidden_size: int, mix_pooled: bool = False):
        super().__init__()

        if mix_pooled:
            self.mixer = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 2 * hidden_size),
                torch.nn.LayerNorm(normalized_shape=[2 * hidden_size]),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Linear(2 * hidden_size, 1)
        else:
            self.mixer = lambda x: x
            self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, pooled_a: torch.Tensor, pooled_b: torch.Tensor) -> torch.Tensor:
        a_val = self.linear(self.mixer(pooled_a))
        b_val = self.linear(self.mixer(pooled_b))
        return b_val - a_val


def get_core_model(model: Union[EncoderDecoderModel, BertLMHeadModel]) -> BertModel:
    if hasattr(model, "encoder"):
        return model.encoder
    else:
        return model.bert


def make_model(
    pretrained_path: str,
    referee_type: str,
    prediction_type: str,
    pooler_type: str,
    model_type: str,
    n_leading_weeks: Optional[int] = None,
    mix_leading_weeks: Optional[bool] = None,
) -> CompetitionModel:
    core_model = get_core_model(
        models.from_pretrained(checkpoint=pretrained_path, model_type=model_type))

    hidden_size = core_model.config.hidden_size

    n_dims_leading_weeks = n_leading_weeks if n_leading_weeks else 0

    if referee_type == "simple":
        referee = SimpleReferee(hidden_size + n_dims_leading_weeks)
    elif referee_type == "triplet":
        referee = TripletReferee(hidden_size + n_dims_leading_weeks)
    elif referee_type == "binary":
        if mix_leading_weeks is None:
            mix_leading_weeks = n_dims_leading_weeks and n_dims_leading_weeks > 0
        referee = BinaryReferee(
            hidden_size + n_dims_leading_weeks,
            mix_pooled=mix_leading_weeks,
        )
    else:
        raise AttributeError("referee_type should be simple/triplet/binary")

    if mix_leading_weeks and referee_type != "binary":
        raise NotImplementedError("Mixing is only implemented for binary referee")

    if model_type == "Decoder":
        pooler = DecoderPooler()
    else:
        if pooler_type == "mean":
            pooler = SimplePooler()
        elif pooler_type == "cls":
            pooler = ClsPooler()
        else:
            raise NotImplemented("Only mean and cls poolers implemented")

    return CompetitionModel(
        core_model,
        pooler,
        referee,
        prediction_type=prediction_type,
    )
