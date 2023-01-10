from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

from app.models.base import Model


@dataclass
class InfoNCELoss(Model):
    temperature: float
    subloss: nn.Module

    def forward(
        self,
        embedding: TensorType["batch_size", "feat_size"],  # noqa: F821
        index: TensorType["batch_size"],  # noqa: F821
    ) -> TensorType[None]:

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

        if isinstance(self.subloss, nn.CrossEntropyLoss):
            return self.subloss(logit, label)

        elif isinstance(self.subloss, nn.MSELoss):
            label_onehot = F.one_hot(label, num_classes=batch_size - 1).to(
                dtype=torch.float
            )
            return self.subloss(logit, label_onehot)

        raise ValueError(
            f"Loss of class `{self.loss.__class__}` not implemented for {self.__class__}!"
        )
