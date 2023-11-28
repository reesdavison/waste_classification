import os
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from skimage import color, filters, io
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2 as v2

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


def mask_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            # converts PIL image to range 0, 1
            transforms.ToTensor(),
            transforms.Resize(
                (32, 32),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
        ]
    )


def get_object_mask(image, light_image, dark_image):
    """Use these images as a threshold to remove bg
    assume image is between 0 and 1
    """
    _image_g = filters.gaussian(color.rgb2gray(image))
    _light_g = filters.gaussian(color.rgb2gray(light_image))
    _dark_g = filters.gaussian(color.rgb2gray(dark_image))
    tol = 0.1
    _2d_mask = (_image_g < (_dark_g - tol)) | (_image_g > (_light_g + tol))
    _2d_mask = filters.median(_2d_mask)
    _3d_mask = np.dstack([_2d_mask, _2d_mask, _2d_mask])
    return _3d_mask


class BagDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir: Path | str, transform=None, remove_bg=False):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.remove_bg = remove_bg
        self._mask_transform = mask_transform()

        # ensure we setup classes in the same order every time
        dirs = sorted(os.listdir(root_dir))

        self.images = []
        self.labels = []
        self.label_map = {}
        self.light_map = {}
        self.dark_map = {}

        for label, dir in enumerate(dirs):
            self.label_map[label] = dir
            self.light_map[label] = Path(dir) / "median_light.jpg"
            self.dark_map[label] = Path(dir) / "median_dark.jpg"
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

    def _get_mask(self, image, label):
        light_img_name = os.path.join(self.root_dir, self.light_map[label])
        dark_img_name = os.path.join(self.root_dir, self.dark_map[label])
        light_image = io.imread(light_img_name)
        dark_image = io.imread(dark_img_name)
        mask = get_object_mask(image, light_image, dark_image)
        return mask

    def _remove_bg(self, image, mask):
        return image * mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]

        img_name = os.path.join(self.root_dir, self.images[idx])
        image = io.imread(img_name)

        mask = self._get_mask(image, label)
        if self.remove_bg:
            image = self._remove_bg(image, mask)

        if self.transform:
            image = self.transform(image)
            mask = self._mask_transform(mask)

        return image, label, mask
