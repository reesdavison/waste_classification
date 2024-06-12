from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

from waste_classification.dataloader import (
    BagDataset,
    basic_transform,
    move_around_transform,
)
from waste_classification.experiment_params import NNClassifierParams, TransformType
from waste_classification.nn.basic_network import Net
from waste_classification.nn.experiment_recorder import ExperimentRecorder
from waste_classification.paths import DATA_DIR, MODEL_DIR, STATS_DIR

torch.manual_seed(0)


def cross_validate(test_loader):
    # perform cross validation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels, _ = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def get_datasets(params: NNClassifierParams, transform_func: callable):
    dataset = BagDataset(
        root_dir=DATA_DIR,
        transform=transform_func(),
        remove_bg=params.remove_bg,
        crop_image=params.crop_image,
    )
    generator1 = torch.Generator().manual_seed(params.split_seed)
    train_set, test_set = torch_data.random_split(
        dataset, [1 - params.test_split, params.test_split], generator=generator1
    )
    return train_set, test_set


if __name__ == "__main__":
    exp_id = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    # Setup model folder
    model_folder = MODEL_DIR / exp_id
    model_folder.mkdir(parents=True, exist_ok=True)

    params = NNClassifierParams(
        id=exp_id,
        num_epochs=50,
        lr=0.001,  # 0.005,
        momentum=0.9,
        batch_size=4,
        split_seed=42,
        test_split=0.2,
        remove_bg=False,
        load_cifar_weights=False,
        transform=TransformType.BASIC,
        crop_image=True,
        # pretrained_model=str(MODEL_DIR / "20231128-23:22:20/final_bag_net.pth"),
        # pretrained_model=str(MODEL_DIR / "20231128-22:06:16/017_bag_net.pth"),
    )

    transform_func = basic_transform
    if params.transform == TransformType.MOVE_AROUND:
        transform_func = move_around_transform

    train_set, test_set = get_datasets(params, transform_func)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=params.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=params.batch_size, shuffle=True, num_workers=2
    )

    net = Net()
    if params.pretrained_model:
        net.load_state_dict(torch.load(params.pretrained_model))
    elif params.load_cifar_weights:
        net.load_state_dict(torch.load("./cifar_net.pth"), strict=False)

    criterion = nn.CrossEntropyLoss()

    # cifar_params lr=0.001, momentum=0.9
    optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=params.momentum)

    # print every print_loss_count mini-batches
    print_loss_count = 5

    recorder = ExperimentRecorder(id=exp_id, stats_dir=STATS_DIR)

    num_iter_per_epoch = len(train_loader)
    params.num_iter_per_epoch = num_iter_per_epoch
    params.data_counts = train_set.dataset.get_label_counts()
    params.write(STATS_DIR / f"{exp_id}_stats.json")

    for epoch in range(params.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, _ = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_loss_count == print_loss_count - 1:
                avg_loss = running_loss / print_loss_count
                iteration = epoch * num_iter_per_epoch + i
                recorder.update(iteration, avg_loss, epoch, accuracy=None)
                print(f"[{epoch}, {i:5d}] loss: {avg_loss:.3f}")
                running_loss = 0.0

        torch.save(net.state_dict(), model_folder / f"{epoch:03}_bag_net.pth")
        accuracy = cross_validate(test_loader)
        recorder.update((epoch + 1) * num_iter_per_epoch, None, epoch, accuracy)
        recorder.save()
        print(f"End of epoch {epoch}: accuracy: {accuracy:.3f}")

    print("Finished Training")
    torch.save(net.state_dict(), model_folder / "final_bag_net.pth")
