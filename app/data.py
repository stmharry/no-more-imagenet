from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import PIL.Image
from pydantic import BaseModel
from torch.utils.data import Dataset
from torchtyping import TensorType

from app.schemas.transforms import Transform


class CheXpertItem(BaseModel):
    images: list[TensorType[1, -1, -1]]

    class Config:
        arbitrary_types_allowed = True


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
            images=[
                self.transform(image),
                self.transform(image),
            ],
        ).dict()

    def __len__(self) -> int:
        return len(self.df)
