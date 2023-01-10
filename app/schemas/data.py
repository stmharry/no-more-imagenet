from enum import Enum

from torch.utils.data import DataLoader, Dataset

from app.schemas.base import ObjectConfig
from app.schemas.transforms import TransformConfig


class DatasetMode(str, Enum):
    TRAIN = "train"
    VAL = "validation"
    DEBUG = "debug"


class DatasetConfig(ObjectConfig[Dataset]):
    csv_path: str
    transform: TransformConfig


class DataLoaderConfig(ObjectConfig[DataLoader]):
    dataset: DatasetConfig
