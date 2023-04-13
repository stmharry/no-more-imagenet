from pathlib import Path
from typing import Mapping

import torch
import yaml
from absl import logging
from pydantic import BaseModel
from torchtyping import TensorType

from app.schemas.base import ObjectConfig
from app.schemas.data import DataLoaderConfig, DatasetMode
from app.schemas.losses import CriterionConfig
from app.schemas.metrics import MetricConfig
from app.schemas.models import ModelConfig

MutableTensorDict = dict[str, torch.Tensor]
TensorDict = Mapping[str, torch.Tensor]
TensorScalar = TensorType[None]  # noqa: F821


class InputFnConfig(BaseModel):
    data_loader: dict[DatasetMode, DataLoaderConfig]


class ModelFnConfig(BaseModel):
    model: dict[str, ModelConfig]
    criterion: dict[str, CriterionConfig]
    metric: dict[str, MetricConfig]


class OptimizerConfig(ObjectConfig[torch.optim.Optimizer]):
    pass


class SchedulerConfig(ObjectConfig[torch.optim.lr_scheduler._LRScheduler]):
    pass


class ScalerConfig(ObjectConfig[torch.cuda.amp.GradScaler]):
    pass


class TrainConfig(BaseModel):
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    scaler: ScalerConfig

    device: str
    use_fp16: bool

    max_steps: int
    log_step_count_steps: int
    save_summary_steps: int
    save_checkpoints_steps: int
    warmup_steps: int = 0


class EstimatorConfig(BaseModel):
    train: TrainConfig


class Config(BaseModel):
    input_fn: InputFnConfig
    model_fn: ModelFnConfig
    estimator: EstimatorConfig

    @classmethod
    def from_path(cls, path: str | Path) -> "Config":
        logging.info(f"Loading config from path {path!s}")

        with open(path, "r") as f:
            obj: dict = yaml.full_load(f)

        return cls.parse_obj(obj=obj)
