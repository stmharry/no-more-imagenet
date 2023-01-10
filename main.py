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
from torchtyping import TensorType

from app.schemas import Config, CriterionConfig, DatasetMode, TrainConfig

flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_string("config", "config.yaml", "Configuration file to use.")
flags.DEFINE_boolean("debug", False, "Debug mode toggle.")
FLAGS = flags.FLAGS


def step(
    data_iter: Iterator,
    models: dict[str, nn.Module],
    criteria: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    config: Config,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:

    criterion_configs: dict[str, CriterionConfig] = config.model_fn.criterion
    train_config: TrainConfig = config.estimator.train
    device = torch.device(train_config.device)
    use_fp16: bool = train_config.use_fp16

    model: nn.Module = models["resnet"]

    (features, labels) = next(data_iter)
    with torch.autocast(train_config.device.split(":")[0], enabled=use_fp16):
        image: TensorType["batch_size", "instances", 1, "width", "height"]  # noqa: F821
        image = features["image"].to(device=device)
        image = torch.broadcast_to(image, image.shape[0:2] + (3,) + image.shape[3:5])
        image = torch.flatten(image, start_dim=0, end_dim=1)

        embedding: TensorType["batch_size", "feat_size"]  # noqa: F821
        embedding = model(image)

        index: TensorType["batch_size", "instances"]  # noqa: F821
        index = labels["index"].to(device=device)
        index = torch.flatten(index, start_dim=0, end_dim=1)

        loss: TensorType[None]  # noqa: F821
        losses: dict[str, TensorType[None]] = {}  # noqa: F821
        for (name, criterion) in criteria.items():
            weight: float = criterion_configs[name].weight

            loss = weight * criterion(embedding, index)
            losses[name] = loss

        total_loss = torch.sum(torch.stack(list(losses.values())))

    optimizer.zero_grad()

    if scaler is not None:
        total_loss = scaler.scale(total_loss)

    total_loss.backward()

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()


def main(_) -> None:
    # processing config

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

    # initializing

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

    models: dict[str, nn.Module] = {}
    for (name, model_config) in config.model_fn.model.items():
        models[name] = model_config.create().to(device=device)

    criteria: dict[str, nn.Module] = {}
    for (name, criterion_config) in config.model_fn.criterion.items():
        criteria[name] = criterion_config.loss.create()

    optimizer = train_config.optimizer.create(params=models["resnet"].parameters())
    scheduler = train_config.scheduler.create(optimizer=optimizer)
    scaler = train_config.scaler.create(enabled=use_fp16)

    for num_step in itertools.count():
        if num_step == train_config.max_steps:
            break

        step(
            data_iter=train_iter,
            models=models,
            criteria=criteria,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
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
