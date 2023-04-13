from dataclasses import dataclass
from typing import TypedDict

import torch
from torchmetrics.functional.classification import multiclass_accuracy
from torchtyping import TensorType

from app.modules import DataclassModule
from app.schemas.core import TensorDict
from app.schemas.losses import TensorLoss


class InfoNCEAccuracyFeaturesInput(TypedDict):
    logit: TensorType["batch_size", "batch_size_m1"]  # noqa: F821


class InfoNCEAccuracyLabelsInput(TypedDict):
    label: TensorType["batch_size"]  # noqa: F821


class InfoNCEAccuracyFeaturesOutput(TypedDict):
    pass


class InfoNCEAccuracyLabelsOutput(TypedDict):
    pass


@dataclass(unsafe_hash=True)
class InfoNCEAccuracy(DataclassModule):
    top_k: int

    def _forward_impl(
        self,
        features: InfoNCEAccuracyFeaturesInput,
        labels: InfoNCEAccuracyLabelsInput,
        stats: TensorDict,
    ) -> tuple[InfoNCEAccuracyFeaturesOutput, InfoNCEAccuracyLabelsOutput, TensorLoss]:

        logit: torch.Tensor = features["logit"]
        label: torch.Tensor = labels["label"]

        num_classes: int = logit.shape[1]

        acc_per_class = multiclass_accuracy(
            preds=logit,
            target=label,
            num_classes=num_classes,
            top_k=self.top_k,
            average=None,
        )

        return ({}, {}, {f"info_nce_acc@{self.top_k}": acc_per_class[0]})
