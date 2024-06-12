import numpy as np
import pytest

from waste_classification.dataloader import BagDataset
from waste_classification.features.features import extract_feature


def test_feature_extraction_bad_size():
    trainset = BagDataset(root_dir="./data")
    train_iter = iter(trainset)
    image, _, mask = next(train_iter)
    with pytest.raises(ValueError, match=r"must be divisible by 3"):
        extract_feature(image, mask, feature_size=10)


def test_feature_extraction():
    trainset = BagDataset(root_dir="./data")
    train_iter = iter(trainset)
    image, _, mask = next(train_iter)
    feature: np.array = extract_feature(image, mask, feature_size=48)
    assert feature.shape == (48,)
