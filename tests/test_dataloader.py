import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torchvision
import torchvision.transforms as transforms
from skimage import draw, io

from recycle_eye.classifier.dataloader import (
    BagDataset,
    basic_transform,
    crop_image_func,
    get_bounding_box_func,
    get_object_mask_func,
    move_around_transform,
)
from recycle_eye.experiment_params import DataCount

"""
Tests:
- Check we can get our dataset input stats to match the CIFAR input stats
- Check basic transforms
- Check background removal transforms
"""

VISUAL_TEST_ENV = "VISUAL_TEST"
VISUAL_TEST_CHECK = "non programmatic visual test check"


def _imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def _raw_imshow(img):
    plt.imshow(img)
    plt.show()


def _image_and_label_checks(images, labels):
    assert isinstance(images, torch.Tensor)
    assert images.size() == torch.Size([4, 3, 32, 32])
    assert images.max() <= torch.Tensor([1.0])
    assert images.min() >= torch.Tensor([-1.0])

    assert isinstance(labels, torch.Tensor)
    assert labels.size() == torch.Size([4])


def test_cifar_dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar_data", train=True, download=False, transform=transform
    )
    assert len(trainset) == 50_000
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    assert len(trainloader) == 50_000 / batch_size

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    _image_and_label_checks(images, labels)


def test_bag_dataset():
    batch_size = 4

    trainset = BagDataset(root_dir="./data", transform=basic_transform())
    num_images = 51 + 51 + 44
    assert len(trainset) == num_images

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    assert len(trainloader) == math.ceil(num_images / batch_size)
    dataiter = iter(trainloader)
    images, labels, _ = next(dataiter)
    _image_and_label_checks(images, labels)
    assert (labels <= 2).all()


def test_get_counts():
    trainset = BagDataset(root_dir="./data", transform=basic_transform())
    assert trainset.get_label_counts() == sorted(
        [
            DataCount(ord_label=0, label="compostable_waste", count=44),
            DataCount(ord_label=1, label="general_waste", count=51),
            DataCount(ord_label=2, label="mixed_recycling", count=51),
        ],
        key=lambda count: count.ord_label,
    )


@pytest.mark.skipif(
    os.environ.get(VISUAL_TEST_ENV) != "remove_bg_single",
    reason=VISUAL_TEST_CHECK,
)
def test_check_remove_bg_single_image():
    dark_name = "./data/compostable_waste/median_dark.jpg"
    light_name = "./data/compostable_waste/median_light.jpg"
    dark_image = io.imread(dark_name)
    light_image = io.imread(light_name)

    image_name = "./data/mixed_recycling/17.jpg"
    image = io.imread(image_name)
    mask = get_object_mask_func(image, light_image, dark_image)
    _raw_imshow(image * mask)


@pytest.mark.skipif(
    os.environ.get(VISUAL_TEST_ENV) != "crop_to_bb_single",
    reason=VISUAL_TEST_CHECK,
)
def test_check_crop_to_bb_single_image():
    dark_name = "./data/compostable_waste/median_dark.jpg"
    light_name = "./data/compostable_waste/median_light.jpg"
    dark_image = io.imread(dark_name)
    light_image = io.imread(light_name)

    image_name = "./data/mixed_recycling/17.jpg"
    image = io.imread(image_name)
    mask = get_object_mask_func(image, light_image, dark_image)
    bb = get_bounding_box_func(mask[:, :, 0])
    rr, cc = draw.rectangle_perimeter(
        start=(bb[0], bb[1]), end=(bb[2], bb[3]), shape=image.shape
    )
    image[rr, cc, :] = 255

    _raw_imshow(image)
    cropped_image = crop_image_func(image, bb)
    _raw_imshow(cropped_image)


@pytest.mark.skipif(
    os.environ.get(VISUAL_TEST_ENV) != "crop_to_bb",
    reason=VISUAL_TEST_CHECK,
)
def test_check_crop_to_bb():
    trainset = BagDataset(
        root_dir="./data", transform=basic_transform(), remove_bg=False, crop_image=True
    )
    for image, _, _ in trainset:
        _imshow(image)


@pytest.mark.skipif(
    os.environ.get(VISUAL_TEST_ENV) != "remove_bg_transform",
    reason=VISUAL_TEST_CHECK,
)
def test_check_remove_bg_transform_images():
    trainset = BagDataset(
        root_dir="./data", transform=basic_transform(), remove_bg=True
    )
    for image, _, _ in trainset:
        _imshow(image)


@pytest.mark.skipif(
    os.environ.get(VISUAL_TEST_ENV) != "basic_transform",
    reason=VISUAL_TEST_CHECK,
)
def test_check_basic_transform_images():
    trainset = BagDataset(root_dir="./data", transform=basic_transform())

    for image, _, _ in trainset:
        _imshow(image)


@pytest.mark.skipif(
    os.environ.get(VISUAL_TEST_ENV) != "move_around_transform",
    reason=VISUAL_TEST_CHECK,
)
def test_check_move_around_transform_images():
    trainset = BagDataset(
        root_dir="./data", transform=move_around_transform(), remove_bg=True
    )

    for image, _, _ in trainset:
        _imshow(image)
