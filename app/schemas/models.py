from typing import Any

from pydantic import root_validator
from torch import nn

from app.schemas.base import ObjectConfig


class ModelConfig(ObjectConfig[nn.Module]):
    @root_validator
    def build_sub_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        field_name: str
        value: Any

        for (field_name, value) in values.items():
            try:
                config: ModelConfig = ModelConfig.parse_obj(value)
            except Exception:
                continue

            values[field_name] = config

        return values
