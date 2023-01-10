from pydantic import BaseModel

from app.schemas.models import ModelConfig


class LossConfig(ModelConfig):
    pass


class CriterionConfig(BaseModel):
    weight: float = 1.0
    loss: LossConfig
