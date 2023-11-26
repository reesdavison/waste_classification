import os
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

# import pandas as pd
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from recycle_eye.classifier.experiment_params import DataCount


def basic_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            # converts PIL image to range 0, 1
            transforms.ToTensor(),
            # converts to range -1, 1
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # pretty drastic resize :fingers_crossed: there's enough information left
            transforms.Resize((32, 32), antialias=True),
        ]
    )


class BagDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir: Path | str, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # ensure we receive in the same order every time
        dirs = sorted(os.listdir(root_dir))

        self.images = []
        self.labels = []
        self.label_map = {}

        for label, dir in enumerate(dirs):
            self.label_map[label] = dir
            for img_name in os.listdir(self.root_dir / dir):
                if re.match(r"\d*\.jpg", img_name):
                    self.images.append(Path(dir) / img_name)
                    self.labels.append(label)

    def get_label_counts(self) -> List[DataCount]:
        return [
            DataCount(
                ord_label=ord_label,
                label=label,
                count=sum([check_label == ord_label for check_label in self.labels]),
            )
            for ord_label, label in self.label_map.items()
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        image = io.imread(img_name)

        if self.transform:
            sample = self.transform(image)

        label = self.labels[idx]
        return sample, label
