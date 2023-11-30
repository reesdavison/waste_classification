from datetime import datetime
from typing import List

import joblib
import numpy as np
import torch
import torch.utils.data as torch_data
from sklearn.neighbors import KNeighborsClassifier

from recycle_eye.dataloader import BagDataset
from recycle_eye.experiment_params import (
    KNNAblationExperiment,
    KNNAlgoType,
    KNNParams,
    KNNResult,
)
from recycle_eye.paths import DATA_DIR, MODEL_DIR, STATS_DIR


def extract_feature(
    image: np.array, mask: np.array, feature_size: int = 48
) -> np.array:
    if feature_size % 3 != 0:
        raise ValueError(f"feature_size {feature_size} must be divisible by 3")
    single_mask = mask[:, :, 0]
    bins_per_dim = feature_size // 3
    feature_vector = np.zeros([bins_per_dim * 3], dtype=np.float32)
    for i in range(3):
        bins, _ = np.histogram(image[:, :, i][single_mask], bins=bins_per_dim)
        normaliser = bins.sum()
        assert normaliser == single_mask.sum(), "Check hist count is same as mask count"
        bins = bins.astype(np.float32)
        bins /= float(normaliser)
        feature_vector[i * bins_per_dim : (i + 1) * bins_per_dim] = bins
    return feature_vector


def extract_all_features(
    dataset: torch_data.Dataset, feature_size: int = 48
) -> (List[np.array], List[int]):
    all_features = []
    all_labels = []

    for data in dataset:
        image, label, mask = data
        feature = extract_feature(image, mask, feature_size=feature_size)
        all_features.append(feature)
        all_labels.append(label)

    return all_features, all_labels


def run_knn_experiment(params: KNNParams) -> KNNResult:
    dataset = BagDataset(root_dir=DATA_DIR)
    generator1 = torch.Generator().manual_seed(params.split_seed)
    train_set, test_set = torch_data.random_split(
        dataset, [1 - params.test_split, params.test_split], generator=generator1
    )

    all_features, all_labels = extract_all_features(train_set, params.feature_size)

    classifier = KNeighborsClassifier(
        n_neighbors=params.n_neighbours, algorithm=params.algorithm.value
    )
    classifier.fit(all_features, all_labels)

    joblib.dump(classifier, MODEL_DIR / f"{params.id}_bag_knn.joblib")

    test_features, gt_labels = extract_all_features(test_set, params.feature_size)
    labels = classifier.predict(test_features)
    # probs = classifier.predict_proba(test_features)

    correct = (gt_labels == labels).sum()
    total = labels.size
    accuracy = correct / total

    return KNNResult(accuracy=accuracy, params=params)


def run_fv_ablation():
    """Test feature vector size

    Not a huge number of vectors so let's stick with brute force for accuracy
    """
    full_exp_id = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    ablation_exp = KNNAblationExperiment(id=full_exp_id, results=[])

    for fv_size in [3, 6, 9, 12, 24, 48]:
        params = KNNParams(
            id=datetime.now().strftime("%Y%m%d-%H:%M:%S"),
            split_seed=42,
            test_split=0.2,
            feature_size=fv_size,
            algorithm=KNNAlgoType.BRUTE,
        )
        result = run_knn_experiment(params)
        print(result)
        ablation_exp.results.append(result)

    ablation_exp.write(STATS_DIR / f"{full_exp_id}.json")


def run_nn_ablation():
    """Test effect of number of neighbours

    Not a huge number of vectors so let's stick with brute force for accuracy
    """
    full_exp_id = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    ablation_exp = KNNAblationExperiment(id=full_exp_id, results=[])

    for num in [1, 2, 3, 4, 5]:
        params = KNNParams(
            id=datetime.now().strftime("%Y%m%d-%H:%M:%S"),
            split_seed=42,
            test_split=0.2,
            feature_size=12,
            algorithm=KNNAlgoType.BRUTE,
            n_neighbours=num,
        )
        result = run_knn_experiment(params)
        print(result)
        ablation_exp.results.append(result)

    ablation_exp.write(STATS_DIR / f"{full_exp_id}.json")


if __name__ == "__main__":
    run_nn_ablation()
    # best result so far with this method
    # {
    #     "accuracy": 0.7586206896551724,
    #     "params": {
    #         "id": "20231128-20:57:32",
    #         "split_seed": 42,
    #         "test_split": 0.2,
    #         "remove_bg": false,
    #         "n_neighbours": 1,
    #         "algorithm": "brute",
    #         "feature_size": 12
    #     }
    # }
