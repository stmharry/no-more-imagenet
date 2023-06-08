# mypy: disable-error-code=name-defined

import itertools
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterator

import torch
from absl import app, flags, logging
from torch import nn
from torch.utils.data import DataLoader
import tensorboardX
from torchtyping import TensorType

from app.schemas import (
    Config,
    CriterionConfig,
    MetricConfig,
    DatasetMode,
    TensorDict,
    TrainConfig,
)


flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_string("config", "config.yaml", "Configuration file to use.")
flags.DEFINE_boolean("debug", False, "Debug mode toggle.")
FLAGS = flags.FLAGS


def step(
    current_step: int,
    data_iter: Iterator,
    models: dict[str, nn.Module],
    criteria: dict[str, nn.Module],
    metrics: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    config: Config,
    scaler: torch.cuda.amp.GradScaler | None = None,
    writer: tensorboardX.SummaryWriter | None = None,
) -> None:
    criterion_configs: dict[str, CriterionConfig] = config.model_fn.criterion
    metrics_configs: dict[str, MetricConfig] = config.model_fn.metric
    train_config: TrainConfig = config.estimator.train
    device = torch.device(train_config.device)
    use_fp16: bool = train_config.use_fp16

    features: TensorDict
    labels: TensorDict
    total_loss: TensorLoss
    (features, labels) = next(data_iter)
    stats = {}

    with torch.autocast(train_config.device.split(":")[0], enabled=use_fp16):
        image: TensorType["batch_size", "instances", 1, "width", "height"]  # noqa: F821
        image = features["image"].to(device=device)
        image = torch.broadcast_to(image, image.shape[0:2] + (3,) + image.shape[3:5])
        image = torch.flatten(image, start_dim=0, end_dim=1)

        features.update({"image": image})

        index: TensorType["batch_size", "instances"]  # noqa: F821
        index = labels["index"].to(device=device)
        index = torch.flatten(index, start_dim=0, end_dim=1)

        labels.update({"index": index})

        # features: exist in both training phase and testing phase
        # labels: only for training phase
        # To Do:
        # 1. remove stats;
        # 2. think about some_list

        # for model in some_list:
        # _features, _labels = model(features, labels)
        # check key collision
        # features.update(_features)
        # labels.update(_labels)

        models["resnet"](features, labels)  # fit data into model
        criteria["info_nce"](features, labels)
        metrics["acc1"](features, labels)
        metrics["acc5"](features, labels)

        total_loss = features["info_nce_loss"]

    optimizer.zero_grad()

    if scaler is None:
        total_loss.backward()
        optimizer.step()

    else:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if writer is not None:
        writer.add_scalar("loss", total_loss, global_step=current_step)


def main(_) -> None:
    # config

    result_dir: Path
    if FLAGS.debug:
        result_dir = Path(tempfile.mkdtemp())

    else:
        result_dir = Path(FLAGS.result_dir, datetime.now().strftime("%Y-%m-%d-%H%M%S"))

    logging.info(f"Making directory {result_dir!s}")
    result_dir.mkdir(parents=True, exist_ok=True)

    config_path: Path = result_dir / "config.yaml"
    shutil.copy(FLAGS.config, config_path)

    config: Config = Config.from_path(config_path)

    # input_fn creation

    data_loader_configs = config.input_fn.data_loader
    train_config = config.estimator.train
    device = torch.device(train_config.device)
    use_fp16 = train_config.use_fp16

    train_loader: DataLoader
    val_loader: DataLoader | None = None
    if FLAGS.debug:
        train_loader = data_loader_configs[DatasetMode.DEBUG].create()

    else:
        train_loader = data_loader_configs[DatasetMode.TRAIN].create()
        data_loader_configs[DatasetMode.VAL].create()

    train_iter: Iterator = iter(train_loader)

    # model_fn creation

    models: dict[str, nn.Module] = {}
    for name, _config in config.model_fn.model.items():
        models[name] = _config.create().to(device=device)

    criteria: dict[str, nn.Module] = {}
    for name, _config in config.model_fn.criterion.items():
        criteria[name] = _config.create()

    metrics: dict[str, nn.Module] = {}
    for name, _config in config.model_fn.metric.items():
        metrics[name] = _config.create()

    optimizer = train_config.optimizer.create(params=models["resnet"].parameters())
    scheduler = train_config.scheduler.create(optimizer=optimizer)
    scaler = train_config.scaler.create(enabled=use_fp16)

    writer = tensorboardX.SummaryWriter(log_dir=result_dir)

    for num_step in itertools.count():
        if num_step == train_config.max_steps:
            break

        step(
            current_step=num_step,
            data_iter=train_iter,
            models=models,
            criteria=criteria,
            metrics=metrics,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            writer=writer,
        )

        if num_step > train_config.warmup_steps:
            scheduler.step()

        do_log: bool = num_step % train_config.log_step_count_steps == 0
        do_summary: bool = num_step % train_config.save_summary_steps == 0

        if do_log or do_summary:
            pass

        if num_step % train_config.log_step_count_steps:
            logging.info(f"Training step {num_step}")


if __name__ == "__main__":
    app.run(main)
