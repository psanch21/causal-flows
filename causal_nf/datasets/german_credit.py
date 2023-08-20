import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset

from causal_nf.distributions import SCM

import numpy as np

import os

import torch
from causal_nf.sem_equations.german_credit import GermanCredit


def compute_split_idx(original_len, split_sizes, k_fold=None):
    all_idx = torch.arange(original_len)
    if len(split_sizes) == 1:
        return [all_idx]
    if isinstance(k_fold, int):
        assert k_fold >= 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        perm = torch.randperm(original_len, generator=generator)
        all_idx = all_idx[perm]
        n = len(perm) * (1 - split_sizes[0])
        all_idx = torch.roll(all_idx, shifts=int(n * k_fold))
    start_idx, end_idx = 0, None
    all_idx_splits = []

    num_splits = len(split_sizes)
    for i, size in enumerate(split_sizes):
        assert isinstance(size, float)
        assert 0 < size
        assert 1 > size
        new_len = int(size * original_len)
        end_idx = new_len + start_idx
        if i == (num_splits - 1):
            all_idx_splits.append(all_idx[start_idx:])
        else:
            all_idx_splits.append(all_idx[start_idx:end_idx])
        start_idx = end_idx

    return all_idx_splits


# %%
class GermanDataset(Dataset):
    def __init__(self, root_dir: str, split: str, seed: int = None):

        self.root_dir = root_dir

        self.seed = seed
        self.split = split

        self.column_names = [
            "Sex",
            "Age",
            "Credit amount",
            "Repayment history",
            "Checking account",
            "Savings",
            "Housing",
        ]
        self.binary_dims = [0, 1, 2, 3, 4, 5, 6]
        self.binary_min_values = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.binary_max_values = torch.tensor(
            [1.0, 500.0, 999999999.0, 99999999.0, 2.0, 4.0, 3.0]
        )
        self.x = None
        self.y = None

        self._add_noise = False

    def set_add_noise(self, value):
        self._add_noise = value

    def _create_data(self):
        """
        This method sets the value for self.X and self.U
        Returns: None

        """
        data_train = np.load(
            os.path.join(self.root_dir, "german_data", "train_1_X.npy")
        )
        data_val = np.load(
            os.path.join(self.root_dir, "german_data", "test_valid_1_X.npy")
        )

        data = torch.tensor(np.concatenate((data_train, data_val), axis=0))
        num_samples = data.shape[0]

        all_idx_splits = compute_split_idx(
            original_len=num_samples, split_sizes=[0.8, 0.1, 0.1], k_fold=self.seed
        )

        if self.split == "train":
            data = data[all_idx_splits[0]]
        elif self.split in ["valid", "val"]:
            data = data[all_idx_splits[1]]
        elif self.split == "test":
            data = data[all_idx_splits[2]]
        else:
            raise NotImplementedError(f"Split {self.split} not implemented")

        y = data[:, -1]
        # Sex, Age, Credit amount, Repayment history, Checking account, Savings, Housing
        x = torch.zeros((data.shape[0], 7))
        x[:, :4] = data[:, :4]
        x[:, 4] = data[:, 4:7].argmax(1)
        x[:, 5] = data[:, 7:12].argmax(1)
        x[:, 6] = data[:, 12:].argmax(1)

        return x, y

    def prepare_data(self) -> None:
        print(f"\nPreparing data...")
        x, y = self._create_data()

        self.x = x
        self.y = y

    def data(self, one_hot=False, scaler=None, x=None):

        if x is not None:
            x_tmp = x.clone()
        else:
            x_tmp = self.x.clone()

        x_output = []
        x_sex = x_tmp[:, [0]]
        x_last = None
        if one_hot:
            x_1, x_2 = x_tmp[:, :4], x_tmp[:, 4:]
            assert len(x_2[:, 0].unique()) == 3
            x_s_1 = torch.nn.functional.one_hot(x_2[:, 0].long(), num_classes=3)
            assert len(x_2[:, 1].unique()) == 5
            x_s_2 = torch.nn.functional.one_hot(x_2[:, 1].long(), num_classes=5)
            assert len(x_2[:, 2].unique()) == 4
            x_s_3 = torch.nn.functional.one_hot(x_2[:, 2].long(), num_classes=4)
            x_last = torch.cat((x_s_1, x_s_2, x_s_3), dim=1).float()
        else:
            x_last = x_tmp[:, -3:]

        if scaler:
            x_norm = scaler.transform(x_tmp)
            x_middle = x_norm[:, 1:4]
        else:
            x_middle = x_tmp[:, 1:4]

        x = torch.cat((x_sex, x_middle, x_last), dim=1)
        return x, self.y

    def __getitem__(self, index):

        x = self.x[index].clone()
        if self._add_noise:
            x[..., self.binary_dims] += torch.rand_like(x[..., self.binary_dims])
        return x, self.y[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        my_str = f"Dataset German Credit\n"
        my_str += f"\tcolumns: {self.column_names}\n"

        return my_str
