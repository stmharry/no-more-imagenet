import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Generic, Type, TypeVar, get_origin, get_type_hints

import yaml
from pydantic import BaseModel, Extra
from torch.utils.data import DataLoader, Dataset

from app.data import DatasetMode
from app.transforms import Transform

T = TypeVar("T")


class ObjectConfig(BaseModel, Generic[T]):
    modules: ClassVar[list[ModuleType]]
    name: str

    @property
    def obj_cls(self) -> Type[T]:
        obj_cls: Type[T] | None

        for module in self.__class__.modules:
            obj_cls = getattr(module, self.name, None)
            if obj_cls is not None:
                break

        if obj_cls is None:
            raise ValueError(
                f"Class {self.name} not found from modules: {self.__class__.modules}!"
            )

        return obj_cls

    def to(self) -> T:
        obj: dict = self.dict(exclude={"name"})

        type_hints: dict[str, Type[Any]] = get_type_hints(self.__class__)
        for (field_name, field_type) in type_hints.items():
            if get_origin(field_type) is ClassVar:
                continue

            field_value: Any = getattr(self, field_name)

            # use `ObjectConfig.to` if field is of type `ObjectConfig`
            if issubclass(field_type, ObjectConfig):
                obj[field_name] = field_value.to()

        return self.obj_cls(**obj)

    class Config:
        extra = Extra.allow


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


class DatasetConfig(ObjectConfig[Dataset]):
    modules: ClassVar[list[ModuleType]] = [importlib.import_module("app.data")]

    csv_path: str
    transform: TransformConfig


class DataLoaderConfig(ObjectConfig[DataLoader]):
    modules: ClassVar[list[ModuleType]] = [
        importlib.import_module("torch.utils.data"),
    ]
    name: str = "DataLoader"

    dataset: DatasetConfig


class Config(BaseModel):
    data_loader: dict[DatasetMode, DataLoaderConfig]

    @classmethod
    def from_path(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            obj: dict = yaml.full_load(f)

        return cls.parse_obj(obj=obj)
