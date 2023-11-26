from pathlib import Path

import pandas as pd


class TrainingStats:
    def __init__(self, stats_dir: Path):
        self.stats_dir = stats_dir
        self.loss_x = []
        self.loss_y = []
        self.epoch = []

    def update(self, loss_x: int, loss_y: float, epoch: int):
        self.loss_x.append(loss_x)
        self.loss_y.append(loss_y)
        self.epoch.append(epoch)

    def save(self):
        df = pd.DataFrame(
            {"iteration": self.loss_x, "avg_loss": self.loss_y, "epoch": self.epoch}
        )
        df.to_csv(stats_dir / "latest_stats.csv")
