from enum import StrEnum
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class OptimiserType(StrEnum):
    SGD = "sgd"


class DataCount(BaseModel):
    ord_label: int = Field(title="Ordinal label")
    label: str
    count: int


class ExperimentParams(BaseModel):
    lr: float = Field(title="Learning rate")
    batch_size: int = Field(title="Batch size")
    optimiser: OptimiserType = Field(title="Optimiser type", default=OptimiserType.SGD)
    momentum: float = Field(title="Momentum")
    num_epochs: int = Field(title="Number of epochs")
    data_counts: List[DataCount] = Field(default=[], title="Data counts")
    num_iter_per_epoch: int = Field(default=-1, title="Number of iterations per epoch")

    def write(self, path: Path):
        with open(path, "w") as fp:
            fp.write(self.model_dump_json(indent=4))
