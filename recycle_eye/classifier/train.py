from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from recycle_eye.classifier.basic_network import Net
from recycle_eye.classifier.dataloader import BagDataset, basic_transform
from recycle_eye.classifier.experiment_params import ExperimentParams
from recycle_eye.classifier.experiment_recorder import ExperimentRecorder

root_dir = Path(__file__).parent.parent.parent
data_dir = root_dir / "data"
model_dir = root_dir / "models"
stats_dir = root_dir / "training_stats"


if __name__ == "__main__":
    exp_id = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    params = ExperimentParams(
        id=exp_id,
        num_epochs=50,
        lr=0.01,
        momentum=0.9,
        batch_size=4,
    )

    trainset = BagDataset(root_dir=data_dir, transform=basic_transform())
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=params.batch_size, shuffle=True, num_workers=2
    )
    net = Net()
    criterion = nn.CrossEntropyLoss()

    # cifar_params lr=0.001, momentum=0.9
    optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=params.momentum)

    # print every print_loss_count mini-batches
    print_loss_count = 5

    recorder = ExperimentRecorder(id=exp_id, stats_dir=stats_dir)

    num_iter_per_epoch = len(trainloader)

    params.num_iter_per_epoch = num_iter_per_epoch
    params.data_counts = trainset.get_label_counts()
    params.write(stats_dir / f"{exp_id}.json")

    for epoch in range(params.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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
                print(f"[{epoch}, {i:5d}] loss: {avg_loss:.3f}")
                iteration = epoch * num_iter_per_epoch + i
                recorder.update(iteration, avg_loss, epoch)
                recorder.save()
                running_loss = 0.0

        torch.save(net.state_dict(), model_dir / f"{epoch:03}_bag_net.pth")

    print("Finished Training")
    torch.save(net.state_dict(), model_dir / "bag_net.pth")
