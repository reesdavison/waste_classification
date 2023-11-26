import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torchvision
import torchvision.transforms as transforms

from recycle_eye.classifier.dataloader import BagDataset, basic_transform
from recycle_eye.classifier.experiment_params import DataCount

"""Let's check we can get our dataset stats to match the CIFAR input data stats to remove potential
bugs in that area before we begin training.
"""


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
    images, labels = next(dataiter)
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


@pytest.mark.skip(reason="Visual test check")
def test_check_some_images():
    trainset = BagDataset(root_dir="./data", transform=basic_transform())

    def _imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    for image, label in trainset:
        _imshow(image)
