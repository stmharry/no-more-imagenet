from typing import Any, Callable

import torch

from app.schemas.base import ObjectConfig

Transform = Callable[[torch.Tensor], torch.Tensor]


class TransformConfig(ObjectConfig[Transform]):
    transforms: list["TransformConfig"] | None = None

    def create(self, **kwargs: dict[str, Any]) -> Transform:
        obj_dict: dict = self.dict()

        if self.transforms is None:
            del obj_dict["transforms"]
        else:
            transforms: list[Transform] = [
                transform.create() for transform in self.transforms
            ]

            obj_dict["transforms"] = transforms

        if kwargs is not None:
            obj_dict.update(kwargs)

        return self.obj_cls(**obj_dict)
