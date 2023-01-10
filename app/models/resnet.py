from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torchtyping import TensorType

from app.models.base import Model


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


@dataclass(unsafe_hash=True)
class ResNetSimCLR(Model):
    backbone: ResNet
    fc0: nn.Module
    relu0: nn.Module
    fc1: nn.Module

    def forward(
        self,
        x: TensorType["batch", "channels", "height", "width"],  # noqa: F821
    ) -> TensorType["batch", "feat_size"]:  # noqa: F821

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

        return x
