from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict

import pandas as pd
import PIL.Image
import torch
from torch.utils.data import Dataset
from torchtyping import TensorType

from app.schemas.transforms import Transform


class CheXpertFeatures(TypedDict):
    image: TensorType["instances", 1, "height", "width"]  # noqa: F821


class CheXpertLabels(TypedDict):
    index: TensorType["instances"]  # noqa: F821


class CheXpertItem(NamedTuple):
    features: CheXpertFeatures
    labels: CheXpertLabels


@dataclass
class CheXpert(Dataset):
    root_dir: str | Path
    csv_path: str | Path
    transform: Transform

    df: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.df: pd.DataFrame = pd.read_csv(self.root_dir / self.csv_path)

    def __getitem__(self, index: int) -> CheXpertItem:
        item: pd.Series = self.df.iloc[index]

        image: PIL.Image.Image = PIL.Image.open(self.root_dir / item.Path)

        return CheXpertItem(
            features=CheXpertFeatures(
                image=torch.stack([self.transform(image), self.transform(image)], dim=0)
            ),
            labels=CheXpertLabels(index=torch.as_tensor([index, index])),
        )

    def __len__(self) -> int:
        return len(self.df)
