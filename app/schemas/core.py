from pathlib import Path

import yaml
from pydantic import BaseModel

from app.schemas.data import DataLoaderConfig, DatasetMode


class Config(BaseModel):
    input_fn: dict[DatasetMode, DataLoaderConfig]

    @classmethod
    def from_path(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            obj: dict = yaml.full_load(f)

        return cls.parse_obj(obj=obj)
