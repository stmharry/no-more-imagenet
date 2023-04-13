from torchtyping import TensorType

from app.schemas.base import ModuleConfig

TensorMetric = TensorType[None]  # noqa: F821


class MetricConfig(ModuleConfig):
    pass
