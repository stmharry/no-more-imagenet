from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

from app.modules import ModelModule
from app.schemas.core import TensorDict
from app.schemas.losses import TensorLoss


class InfoNCELossFeaturesInput(TypedDict):
    embedding: TensorType["batch_size", "feat_size"]  # noqa: F821


class InfoNCELossLabelsInput(TypedDict):
    index: TensorType["batch_size"]  # noqa: F821


class InfoNCELossFeaturesOutput(TypedDict):
    logit: TensorType["batch_size", "batch_size_m1"]  # noqa: F821


class InfoNCELossLabelsOutput(TypedDict):
    label: TensorType["batch_size"]  # noqa: F821


@dataclass(unsafe_hash=True)
class InfoNCELoss(ModelModule):
    weight: float
    temperature: float
    f: nn.Module

    def _forward_impl(
        self,
        features: InfoNCELossFeaturesInput,
        labels: InfoNCELossLabelsInput,
        stats: TensorDict,
    ) -> tuple[InfoNCELossFeaturesOutput, InfoNCELossLabelsOutput, TensorLoss]:
        embedding: torch.Tensor = features["embedding"]
        index: torch.Tensor = labels["index"]

        batch_size: int = embedding.shape[0]
        device: torch.device = embedding.device

        embedding = F.normalize(embedding, dim=1)
        similarity_pred = torch.matmul(embedding, embedding.T)

        similarity_true = index[None, :] == index[:, None]
        similarity_true = similarity_true.to(device=device)

        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity_pred = similarity_pred[~mask].view(batch_size, -1)
        similarity_true = similarity_true[~mask].view(batch_size, -1)

        positive = similarity_pred[similarity_true].view(batch_size, -1)
        negative = similarity_pred[~similarity_true].view(batch_size, -1)
        logit = torch.cat([positive, negative], dim=1)
        logit = logit / self.temperature

        label = torch.zeros(batch_size, dtype=torch.long, device=device)

        if isinstance(self.f, nn.CrossEntropyLoss):
            loss = self.f(logit, label)

        elif isinstance(self.f, nn.MSELoss):
            label_onehot = F.one_hot(label, num_classes=batch_size - 1).to(
                dtype=torch.float
            )
            loss = self.f(logit, label_onehot)

        else:
            raise ValueError(
                f"Loss of class `{self.loss.__class__}` not implemented for {self.__class__}!"
            )

        return (
            {"logit": logit},
            {"label": label},
            {"info_nce_loss": self.weight * loss},
        )
