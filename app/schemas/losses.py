from torchtyping import TensorType

from app.schemas.base import ModuleConfig

TensorLoss = TensorType[None]  # noqa: F821


class CriterionConfig(ModuleConfig):
    weight: float = 1.0
