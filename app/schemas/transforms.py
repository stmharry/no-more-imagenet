import importlib
from types import ModuleType
from typing import Callable, ClassVar

import torch

from app.schemas.base import ObjectConfig

Transform = Callable[[torch.Tensor], torch.Tensor]


class TransformConfig(ObjectConfig[Transform]):
    modules: ClassVar[list[ModuleType]] = [
        importlib.import_module("torchvision.transforms"),
        importlib.import_module("app.transforms"),
    ]

    transforms: list["TransformConfig"] | None = None

    def to(self) -> Transform:
        obj: dict = self.dict(exclude={"name"})

        if self.transforms is None:
            del obj["transforms"]

        else:
            transforms: list[Transform] = [
                transform.to() for transform in self.transforms
            ]

            obj["transforms"] = transforms

        return self.obj_cls(**obj)
