import abc
from enum import Enum
from typing import Iterable, Type, get_type_hints

from absl import logging
from torch import nn

from app.schemas.core import MutableTensorDict, TensorDict


class TensorDictType(Enum):
    FEATURES = "features"
    LABELS = "labels"


class BaseModule(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        nn.Module.__init__(instance)

        return instance

    def _update_arg_dict(
        self,
        input_dict: MutableTensorDict,
        update_dict: TensorDict,
        type_: TensorDictType,
    ) -> None:
        for key in update_dict.keys():
            if key in input_dict:
                raise ValueError(f"Key `{key}` already present in the state dict!")

        keys: list[str] = list(update_dict.keys())

        logging.info(
            f"Module `{self.__class__.__name__}` producing state dict `{type_.value}` "
            f"with fields {keys}."
        )
        input_dict.update(update_dict)

    def _build_arg_dict(
        self, input_dict: MutableTensorDict, type_: TensorDictType
    ) -> TensorDict:
        type_hint = get_type_hints(self._forward_impl)[type_.value]

        cls_: Type
        keys: Iterable[str]
        if type_hint is TensorDict:
            cls_ = dict
            keys = []
        else:
            cls_ = type_hint
            keys = type_hint.__annotations__.keys()

        build_dict: MutableTensorDict = {}
        for key in keys:
            if key not in input_dict:
                logging.fatal(
                    f"Key `{key}` not present in state dict `{type_.value}`! "
                    f"Available keys are: {list(input_dict)}."
                )

            build_dict[key] = input_dict[key]

        return cls_(**build_dict)

    @abc.abstractmethod
    def _forward_impl(self) -> tuple:
        ...


class ModelModule(BaseModule):
    """ModelMoudle could be Model, Loss, or Metric"""

    @abc.abstractmethod
    def _forward_impl(
        self, features: TensorDict, labels: TensorDict
    ) -> tuple[TensorDict, TensorDict]:
        """purpose?"""
        pass

    def forward(
        self,
        features: MutableTensorDict,
        labels: MutableTensorDict,
    ) -> tuple[MutableTensorDict, MutableTensorDict]:
        """this update function used in the main.step() during training"""
        (_features, _labels) = self._forward_impl(
            features=self._build_arg_dict(features, TensorDictType.FEATURES),
            labels=self._build_arg_dict(labels, TensorDictType.LABELS),
        )

        self._update_arg_dict(features, _features, type_=TensorDictType.FEATURES)
        self._update_arg_dict(labels, _labels, type_=TensorDictType.LABELS)

        return (features, labels)
