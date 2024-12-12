from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo


class StatlogDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Loads the Covertype dataset as a pytorch Dataset from the UCI repository (https://archive.ics.uci.edu/dataset/98/statlog+project).

    Args:
        root (str): Where to store the dataset
        download (bool): Whether to download the dataset
    """

    num_actions: int = 7
    context_size: int = 20
    num_samples: int = 1000

    def __init__(self, root: str = "./data", download: bool = True):
        self.data = fetch_ucirepo(id=144)
        self.X = self.data.data.features.astype(np.float32)
        self.y = self.data.data.targets.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.zeros(self.num_actions, dtype=torch.float32)
        y_item[self.y[idx]] = 1.0

        return X_item, y_item
