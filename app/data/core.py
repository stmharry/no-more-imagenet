from dataclasses import dataclass
from typing import Any, Iterator

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter


@dataclass
class CyclingIterator(_BaseDataLoaderIter):
    loader: DataLoader

    def __iter__(self) -> "CyclingIterator":
        return self

    def __next__(self):
        iterator: Iterator[Any]

        iterator = DataLoader.__iter__(self.loader)
        while True:
            try:
                return next(iterator)
            except StopIteration:
                iterator = DataLoader.__iter__(self.loader)


class CyclingDataLoader(DataLoader):
    def __iter__(self) -> CyclingIterator:
        return CyclingIterator(loader=self)
