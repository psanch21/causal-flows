import os.path
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm
import wandb
from torchlikelihoods import (
    scalers_dict,
    HeterogeneousScaler,
    HeterogeneousObjectScaler,
)
from torchvision.utils import save_image

import causal_nf.utils.io as causal_io
import causal_nf.utils.wandb_local as wandb_local


class BaseDatasetPreparator(ABC):
    def __init__(
        self,
        name,
        splits,
        shuffle_train,
        single_split,
        task,
        k_fold,
        root,
        loss,
        scale,
        device="cpu",
    ):
        self.name = name

        self.split = splits
        self.split_names = ["train", "val", "test"]
        self.current_split = None
        assert sum(splits) == 1
        self.shuffle_train = shuffle_train
        self.single_split = single_split
        self.task = task
        self.k_fold = k_fold

        self.scale = scale
        self.scaler = None
        self.loss = self._loss(loss)

        self.datasets = None

        self.device = device

        if not os.path.exists(root):
            causal_io.makedirs(root)

        self.root = root

    @classmethod
    def params(cls, dataset):

        if isinstance(dataset, dict):
            dataset = causal_io.dict_to_cn(dataset)

        return {
            "splits": dataset.splits,
            "k_fold": dataset.k_fold,
            "shuffle_train": dataset.shuffle_train,
            "single_split": dataset.single_split,
            "loss": dataset.loss,
            "root": dataset.root,
            "scale": dataset.scale,
        }

    @property
    def type_of_data(self):
        return "default"

    # Abstract methods
    @abstractmethod
    def _batch_element_list(self):
        pass

    @abstractmethod
    def _data_loader(self, dataset, batch_size, shuffle, num_workers=0):
        pass

    @abstractmethod
    def _get_dataset_raw(self):
        pass

    @abstractmethod
    def _metric_names(self):
        pass

    @abstractmethod
    def _plot_data(self, batch, **kwargs):
        pass

    @abstractmethod
    def _split_dataset(self, dataset_raw):
        pass

    @abstractmethod
    def _x_dim(self):
        pass

    @abstractmethod
    def get_dataset_train(self):
        pass

    @abstractmethod
    def get_features_train(self):
        pass

    @abstractmethod
    def get_scaler_info(self):
        pass

    @abstractmethod
    def _loss(self, loss):
        pass

    # Not implemented methods

    def monitor(self):
        return "val_loss", "min"

    # Implemented methods

    def non_linearity(self):

        scaler = self._get_scaler()
        if scaler is None:
            return nn.Identity()
        else:
            return scaler.non_linearity()

    def get_batch_elements(self, batch, elements):
        batch_out = []

        batch_elements = self._batch_element_list()

        for el in elements:
            batch_out.append(batch[batch_elements.index(el)])

        return batch_out

    @torch.no_grad()
    def on_start(self, device):
        self.device = device

    def _transform_dataset_pre_split(self, dataset_raw):
        return dataset_raw

    def x_dim(self):
        return self._x_dim()

    def prepare_data(self):
        dataset_raw = self._get_dataset_raw()
        dataset_raw = self._transform_dataset_pre_split(dataset_raw=dataset_raw)
        datasets = self._split_dataset(dataset_raw)
        if self.single_split in self.split_names:
            idx = self.split_names.index(self.single_split)
            for i in range(len(datasets)):
                if i != idx:
                    datasets[i] = datasets[idx]
        datasets = self._transform_after_split(datasets)
        self.datasets = datasets
        return

    def set_current_split(self, i):
        if isinstance(self.single_split, str):
            self.current_split = self.single_split
        else:
            self.current_split = self.split_names[i]

    def _transform_after_split(self, datasets):

        return datasets

    def _get_scaler_list(self, scalers_info):

        if len(scalers_info) == 1:
            name, _ = scalers_info[0]
            scaler = scalers_dict[name]()
        else:
            scalers, splits = [], []
            for (name, size) in scalers_info:
                scaler = scalers_dict[name]()
                scalers.append(scaler)
                splits.append(size)
            scaler = HeterogeneousScaler(scalers, splits)
        return scaler

    def _get_scaler(self):
        if isinstance(self.scale, str):
            scalers_info = self.get_scaler_info()
        else:
            scalers_info = None

        if isinstance(scalers_info, list):
            scaler = self._get_scaler_list(scalers_info=scalers_info)

        elif isinstance(scalers_info, dict):
            sca_info = {}
            for attr_name, value in scalers_info.items():
                if isinstance(value, str):
                    sca_info[attr_name] = value  # Reuse scaler from attribute value
                elif isinstance(value, list):
                    sca_info[attr_name] = self._get_scaler_list(scalers_info=value)
                else:
                    scaler_name = value[0]
                    domain_size = value[1]
                    sca_info[attr_name] = scalers_dict[scaler_name]()
            scaler = HeterogeneousObjectScaler(scalers_dict=sca_info)

        else:
            scaler = scalers_dict["identity"]()

        return scaler

    def get_scaler(self, fit=True):

        scaler = self._get_scaler()

        if fit:
            x = self.get_features_train()
            scaler.fit(x, dims=self.dims_scaler)

        self.scaler = scaler

        return scaler

    @property
    def dims_scaler(self):
        return None  # single scalar per scaler parameter

    def compute_metrics(self, **kwargs):
        return {}

        predictions_dict = {}
        targets_dict = {}

        for name, values in kwargs.items():
            if "logits" in name:
                pred = logits_to_hard_pred_fn(values)
                predictions_dict[name.replace("logits", "")] = pred
            if "target" in name:
                targets_dict[name.replace("target", "")] = values

        preds_target_provided = len(predictions_dict) > 0 and len(targets_dict) > 0
        metric_names = self._metric_names()
        metrics = {}

        if "mae" in metric_names and preds_target_provided:
            raise NotImplementedError
            mae = tm.MeanAbsoluteError()
            value = mae(preds=logits, target=target)
            metrics["mae"] = value
            if "logits_recursive" in kwargs:
                logits_recursive = kwargs["logits_recursive"]
                target_recursive = kwargs["target_recursive"]
                value_2 = mae(preds=logits_recursive, target=target_recursive)
                metrics["mae_recursive"] = value_2

        metrics = {key: value.item() for key, value in metrics.items()}
        return metrics

    def get_ckpt_name(self, ckpt_file):
        ckpt_name = os.path.splitext(os.path.basename(ckpt_file))[0]
        if "epoch" in ckpt_name:
            ckpt_dict = wandb_local.str_to_dict(
                my_str=ckpt_name, sep="-", remove_ext=False
            )
            ckpt_name = f"ckpt_{ckpt_dict['epoch']}"

        return ckpt_name

    def get_dataloader_train(self, batch_size, num_workers=0, shuffle=None):
        assert isinstance(self.datasets, list)

        dataset = self.datasets[0]
        shuffle = self.shuffle_train if shuffle is None else shuffle
        loader_train = self._data_loader(
            dataset, batch_size, shuffle=shuffle, num_workers=num_workers
        )

        return loader_train

    def get_dataloaders(self, batch_size, num_workers=0):
        assert isinstance(self.datasets, list)
        loader_train = self.get_dataloader_train(batch_size, num_workers)

        loaders = [loader_train]
        for i in range(1, len(self.datasets)):
            dataset = self.datasets[i]
            loader = self._data_loader(
                dataset, batch_size, shuffle=False, num_workers=num_workers
            )
            loaders.append(loader)
        return loaders

    def plot_data(
        self,
        split="train",
        num_samples=1,
        shuffle=False,
        batch_idx=0,
        folder=None,
        filename=None,
        show=False,
        **kwargs,
    ):

        loader = self._data_loader(
            dataset=self.datasets[self.split_names.index(split)],
            batch_size=num_samples,
            shuffle=shuffle,
            num_workers=0,
        )

        for i, batch in enumerate(iter(loader)):
            if i == batch_idx:
                break

        return self.plot_data_batch(
            batch, folder=folder, filename=filename, show=show, **kwargs
        )

    def plot_data_batch(self, batch, folder=None, filename=None, show=False, **kwargs):

        fig = self._plot_data(batch, **kwargs)
        if not isinstance(fig, torch.Tensor):
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        self.post_plotting(fig, folder=folder, filename=filename, show=show)

        return fig

    def post_plotting(self, fig, folder=None, filename=None, show=False):
        plt.tight_layout()
        if folder and filename:
            self.save_fig(fig, folder, filename)
        if show:
            plt.show()
        else:
            plt.close("all")

    def get_number_of_rows_and_cols(self, num_samples, batch_size):
        if isinstance(batch_size, tuple):
            bath_size_ = np.prod(batch_size)
            num_samples = min(num_samples, bath_size_)
            ncol = batch_size[1]
            nrow = batch_size[0]
        elif isinstance(batch_size, int):
            num_samples = min(num_samples, batch_size)
            ncol = int(np.ceil(np.sqrt(num_samples)))
            nrow = 1 if num_samples == 2 else ncol
        else:
            ncol = int(np.ceil(np.sqrt(num_samples)))
            nrow = 1 if num_samples == 2 else ncol
        return num_samples, nrow, ncol

    def save_fig(self, fig, folder, filename):
        my_dict = wandb_local.str_to_dict(my_str=filename, sep="--", remove_ext=True)
        name = wandb_local.dict_to_str(my_dict=my_dict, keys_remove=["epoch", "now"])

        full_filaname = os.path.join(folder, f"{filename}")

        if os.path.exists(full_filaname):
            causal_io.print_warning(f"Overwriting file: {full_filaname}")
        if isinstance(fig, torch.Tensor):
            save_image(fig, full_filaname)
            try:
                image = wandb.Image(fig)
                wandb.log({f"figures/{name}": image})
            except:
                causal_io.print_warning(f"wandb not ready to plot image")
        else:
            try:
                wandb.log({f"figures/{name}": wandb.Image(fig)})
            except:
                causal_io.print_warning(f"wandb not ready to plot figure")
            fig.savefig(full_filaname)

    def add_title(self, title_el, ax):

        if title_el.ndim == 0:
            my_str = str(title_el)
            num_characters = 0
        else:
            my_str = ", ".join([f"{t:.2f}" for t in title_el])
            my_str = f"[{my_str}]"
            num_characters = len(my_str)
        fontsize = max(14 - num_characters, 5)
        ax.set_title(f"{my_str}", fontsize=fontsize)

    def select_axis(self, nrow, ncol, i, j, axes):
        if nrow == 1 and ncol == 1:
            ax_ij = axes
        elif nrow == 1:
            ax_ij = axes[j]
        else:
            ax_ij = axes[i, j]

        return ax_ij
