from enum import StrEnum
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class OptimiserType(StrEnum):
    SGD = "sgd"


class TransformType(StrEnum):
    BASIC = "basic"
    AUTO_AUGMENT = "auto_augment"
    MOVE_AROUND = "move_around"


class DataCount(BaseModel):
    ord_label: int = Field(title="Ordinal label")
    label: str
    count: int


class KNNAlgoType(StrEnum):
    AUTO = "auto"
    BALL_TREE = "ball_tree"
    KD_TREE = "kd_tree"
    BRUTE = "brute"


class WriteableBaseModel(BaseModel):
    pass

    def write(self, path: Path):
        with open(path, "w") as fp:
            fp.write(self.model_dump_json(indent=4))


class BaseExperimentParams(WriteableBaseModel):
    id: str = Field(title="Experiment ID")
    split_seed: int = Field(default=42, title="Seed for splitting the data")
    test_split: float = Field(default=0.2, title="Ratio of data in the test set")
    remove_bg: bool = Field(default=False, title="Remove background of images")


class NNClassifierParams(BaseExperimentParams):
    lr: float = Field(default=0.01, title="Learning rate")
    batch_size: int = Field(default=4, title="Batch size")
    optimiser: OptimiserType = Field(title="Optimiser type", default=OptimiserType.SGD)
    momentum: float = Field(default=0.9, title="Momentum")
    num_epochs: int = Field(default=50, title="Number of epochs")
    data_counts: List[DataCount] = Field(default=[], title="Data counts")
    num_iter_per_epoch: int = Field(default=-1, title="Number of iterations per epoch")
    split_seed: int = Field(default=42, title="Seed for splitting the data")
    test_split: float = Field(default=0.2, title="Ratio of data in the test set")
    remove_bg: bool = Field(default=False, title="Remove background of images")
    load_cifar_weights: bool = Field(
        default=False, title="Load weights from pretrained cifar model"
    )
    transform: TransformType = Field(
        default=TransformType.BASIC,
        description="Type of transform to apply to image once bg removed or not removed.",
    )
    pretrained_model: str | None = Field(
        default=None,
        title="Load in pretrained model",
        description="Load model weights in as the starting point for training.",
    )
    crop_image: bool = Field(default=False, title="Crop the image to its bounding box")


class KNNParams(BaseExperimentParams):
    n_neighbours: int = Field(default=5, title="Number of neighbours in KNN classifier")
    algorithm: KNNAlgoType = Field(
        default=KNNAlgoType.AUTO, title="KNN Algorithm to use"
    )
    feature_size: int = Field(
        default=48, title="Size of feature vector, must be divisible by 3"
    )


class KNNResult(BaseModel):
    accuracy: float
    params: KNNParams = Field(title="Params for the experiment")


class KNNAblationExperiment(WriteableBaseModel):
    id: str = Field(title="Experiment ID")
    results: List[KNNResult] = Field(
        description="List of results over different params"
    )
