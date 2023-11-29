import json
from pathlib import Path

import pandas as pd

from recycle_eye.experiment_params import NNClassifierParams


def get_nn_run(exp_id: str, root_folder: str) -> (pd.DataFrame, NNClassifierParams):
    root_folder = Path(root_folder)
    df = pd.read_csv(root_folder / f"{exp_id}.csv", index_col=0)
    with open(root_folder / f"{exp_id}_stats.json", "r") as fp:
        params = NNClassifierParams(**json.load(fp))

    return df, params


def get_accuracy_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.accuracy.isna()][["epoch", "accuracy"]]


def get_loss_df(df: pd.DataFrame, params: NNClassifierParams) -> pd.DataFrame:
    loss_df = df[df.accuracy.isna()][["iteration", "epoch", "avg_loss"]]
    iter_above_epoch = (
        loss_df.iteration % params.num_iter_per_epoch / params.num_iter_per_epoch
    )
    loss_df["cont_epoch"] = loss_df.epoch + iter_above_epoch
    return loss_df
