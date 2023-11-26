from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from recycle_eye.classifier.basic_network import Net
from recycle_eye.classifier.dataloader import BagDataset, basic_transform
from recycle_eye.classifier.stats import TrainingStats

batch_size = 4

root_dir = Path(__file__).parent.parent.parent
data_dir = root_dir / "data"
model_dir = root_dir / "models"
stats_dir = root_dir / "training_stats"

trainset = BagDataset(root_dir=data_dir, transform=basic_transform())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

if __name__ == "__main__":
    net = Net()
    criterion = nn.CrossEntropyLoss()

    # cifar_params lr=0.001, momentum=0.9
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # print every print_loss_count mini-batches
    print_loss_count = 5

    stats = TrainingStats(stats_dir=stats_dir)

    epoch_len = len(trainloader)

    for epoch in range(50):  # loop over the dataset multiple times
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
                iteration = epoch * epoch_len + i
                stats.update(iteration, avg_loss, epoch)
                stats.save()
                running_loss = 0.0

        torch.save(net.state_dict(), model_dir / f"{epoch:03}_bag_net.pth")

    print("Finished Training")
    torch.save(net.state_dict(), model_dir / "bag_net.pth")
