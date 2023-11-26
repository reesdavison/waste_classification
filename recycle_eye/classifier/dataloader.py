import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# import pandas as pd
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


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

        dirs = os.listdir(root_dir)

        self.images = []
        self.labels = []
        self.label_map = {}

        for label, dir in enumerate(dirs):
            self.label_map[label] = dir
            for img_name in os.listdir(self.root_dir / dir):
                if re.match(r"\d*\.jpg", img_name):
                    self.images.append(Path(dir) / img_name)
                    self.labels.append(label)

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
