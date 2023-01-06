import importlib
from enum import Enum
from types import ModuleType
from typing import ClassVar

from torch.utils.data import DataLoader, Dataset

from app.schemas.base import ObjectConfig
from app.schemas.transforms import TransformConfig


class DatasetMode(str, Enum):
    TRAIN = "train"
    VAL = "validation"
    DEBUG = "debug"


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
