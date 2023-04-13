from dataclasses import dataclass

import torch
from torch import nn

from app.models.base import Model
from app.schemas.core import TensorDict


@dataclass(unsafe_hash=True)
class Metric(Model):
    f: nn.Module
    args: list[str]
    kwargs: dict[str, str]

    def _get_value(
        self, key: str, features: TensorDict, labels: TensorDict
    ) -> torch.Tensor:
        (left, _, right) = key.partition("/")

        values: TensorDict
        match left:
            case "features":
                values = features
            case "labels":
                values = labels
            case _:
                raise ValueError(
                    f"Metric input key {arg} has invalid reference {left}!"
                )

        if right not in values:
            raise ValueError(f"Metric input key {arg} has invalid reference {right}!")

        value: torch.Tensor = values[right]

    def forward(
        self, features: TensorDict, labels: TensorDict
    ) -> tuple[TensorDict, TensorDict,]:

        args: list[torch.Tensor] = []
        for arg in self.args:
            args.append(self._get_value(arg, features, labels))

        kwargs: dict[str, torch.Tensor] = {}
        for (key, arg) in self.kwargs.items():
            kwargs[key] = self._get_value(arg, features, labels)

        self.f(*args, **kwargs)
        breakpoint()
