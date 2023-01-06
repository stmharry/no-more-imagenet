from typing import Callable

from absl import app, flags
from torch.utils.data import DataLoader

from app.config import Config
from app.data import DatasetMode

flags.DEFINE_string("config", "config.yaml", "Configuration file to use.")
flags.DEFINE_enum("run", "train", ["train", "debug"], "Run endpoint.")
FLAGS = flags.FLAGS


def train(config: Config):
    train_loader: DataLoader = config.input_fn[DatasetMode.TRAIN].to()
    val_loader: DataLoader = config.input_fn[DatasetMode.VAL].to()

    # TODO(stmharry)


def debug(config: Config):
    debug_loader: DataLoader = config.input_fn[DatasetMode.DEBUG].to()

    for batch in debug_loader:
        print(batch)
        break

    # TODO(stmharry)


def main(_):
    config: Config = Config.from_path(FLAGS.config)

    fn: Callable | None = globals().get(FLAGS.run)
    if fn is None:
        raise ValueError(f"No endpoint defined for `run` {FLAGS.run}!")

    return fn(config=config)


if __name__ == "__main__":
    app.run(main)
