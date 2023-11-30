from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from recycle_eye.classifier.basic_network import Net
from recycle_eye.classifier.dataloader import BagDataset, basic_transform
from recycle_eye.classifier.experiment_recorder import TrainingStats

batch_size = 4

root_dir = Path(__file__).parent.parent.parent
data_dir = root_dir / "data"
model_dir = root_dir / "models"
stats_dir = root_dir / "training_stats"

trainset = BagDataset(root_dir=data_dir, transform=basic_transform())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

model_path = "foo"

net = Net()
net.load_state_dict(torch.load(model_path))
