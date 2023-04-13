from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch
from torch import nn
from torchtyping import TensorType

from app.modules import DataclassModule
from app.schemas.core import TensorDict


class ResNet(Protocol):
    conv1: nn.Module
    bn1: nn.Module
    relu: nn.Module
    maxpool: nn.Module
    layer1: nn.Module
    layer2: nn.Module
    layer3: nn.Module
    layer4: nn.Module
    avgpool: nn.Module


class ResNetSimCLRFeaturesInput(TypedDict):
    image: TensorType["batch", "channels", "height", "width"]  # noqa: F821


class ResNetSimCLRFeaturesOutput(TypedDict):
    embedding: TensorType["batch", "feat_size"]  # noqa: F821


@dataclass(unsafe_hash=True)
class ResNetSimCLR(DataclassModule):
    backbone: ResNet
    fc0: nn.Module
    relu0: nn.Module
    fc1: nn.Module

    def _forward_impl(
        self, features: ResNetSimCLRFeaturesInput, labels: TensorDict, stats: TensorDict
    ) -> tuple[ResNetSimCLRFeaturesOutput, TensorDict, TensorDict]:

        x: torch.Tensor = features["image"]

        backbone: ResNet = self.backbone
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)

        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc0(x)
        x = self.relu0(x)
        x = self.fc1(x)

        return ({"embedding": x}, {}, {})
