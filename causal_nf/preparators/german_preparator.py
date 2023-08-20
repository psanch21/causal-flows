from causal_nf.datasets.german_credit import GermanDataset
from causal_nf.distributions.scm import SCM
from causal_nf.preparators.scm._base_distributions import pu_dict
from causal_nf.preparators.tabular_preparator import TabularPreparator
from causal_nf.sem_equations import sem_dict
from causal_nf.transforms import CausalEquations
from causal_nf.utils.io import dict_to_cn

from causal_nf.utils.scalers import StandardTransform

import numpy as np

import networkx as nx
from torch.distributions import Independent, Normal, Uniform, Laplace
import torch


class GermanPreparator(TabularPreparator):
    def __init__(self, add_noise, **kwargs):

        self.dataset = None
        self.add_noise = add_noise
        sem_fn = sem_dict["german"](sem_name="dummy")

        self.adjacency = sem_fn.adjacency

        self.num_nodes = len(sem_fn.functions)

        self.intervention_index_list = sem_fn.intervention_index_list()
        super().__init__(name="german", task="modeling", **kwargs)

        assert self.split == [0.8, 0.1, 0.1]

    @classmethod
    def params(cls, dataset):
        if isinstance(dataset, dict):
            dataset = dict_to_cn(dataset)

        my_dict = {
            "add_noise": dataset.add_noise,
        }

        my_dict.update(TabularPreparator.params(dataset))

        return my_dict

    @classmethod
    def loader(cls, dataset):
        my_dict = GermanPreparator.params(dataset)

        return cls(**my_dict)

    def _x_dim(self):
        return self.num_nodes

    def get_intervention_list(self):
        x = self.get_features_train().numpy()

        perc_idx = [25, 50, 75]

        percentiles = np.percentile(x, perc_idx, axis=0)
        int_list = []
        for i in self.intervention_index_list:
            percentiles_i = percentiles[:, i]
            values_i = []
            for perc_name, perc_value in zip(perc_idx, percentiles_i):
                values_i.append({"name": f"{perc_name}p", "value": perc_value})

            for value in values_i:
                value["value"] = round(value["value"], 2)
                value["index"] = i
                int_list.append(value)

        return int_list

    def diameter(self):
        adjacency = self.adjacency(True).numpy()
        G = nx.from_numpy_matrix(adjacency, create_using=nx.Graph)
        diameter = nx.diameter(G)
        return diameter

    def longest_path_length(self):
        adjacency = self.adjacency(False).numpy()
        G = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph)
        longest_path_length = nx.algorithms.dag.dag_longest_path_length(G)
        return int(longest_path_length)

    def get_ate_list(self):
        x = self.get_features_train().numpy()

        perc_idx = [25, 50, 75]

        percentiles = np.percentile(x, perc_idx, axis=0)
        int_list = []
        for i in self.intervention_index_list:
            percentiles_i = percentiles[:, i]
            values_i = []
            values_i.append(
                {"name": "25_50", "a": percentiles_i[0], "b": percentiles_i[1]}
            )
            values_i.append(
                {"name": "25_75", "a": percentiles_i[0], "b": percentiles_i[2]}
            )
            values_i.append(
                {"name": "50_75", "a": percentiles_i[1], "b": percentiles_i[2]}
            )
            for value in values_i:
                value["a"] = round(value["a"], 2)
                value["b"] = round(value["b"], 2)
                value["index"] = i
                int_list.append(value)

        return int_list

    def get_ate_list_2(self):
        x = self.get_features_train()

        x_mean = x.mean(0)
        x_std = x.std(0)
        int_list = []
        for i in self.intervention_index_list:
            x_mean_i = x_mean[i].item()
            x_std_i = x_std[i].item()
            values_i = []
            values_i.append({"name": "mu_std", "a": x_mean_i, "b": x_mean_i + x_std_i})
            values_i.append({"name": "mu_-std", "a": x_mean_i, "b": x_mean_i - x_std_i})
            values_i.append(
                {"name": "-std_std", "a": x_mean_i - x_std_i, "b": x_mean_i + x_std_i}
            )
            for value in values_i:
                value["a"] = round(value["a"], 2)
                value["b"] = round(value["b"], 2)
                value["index"] = i
                int_list.append(value)

        return int_list

    def intervene(self, index, value, shape):
        if len(shape) == 1:
            shape = (shape[0], 7)

        x = self.get_features_train()
        cond = x[..., index].floor() == int(value)
        x = x[cond, :]

        return x[: shape[0]]

    def compute_ate(self, index, a, b, num_samples=10000):
        ate = torch.rand((6)) * 2 - 1.0
        return ate

    def compute_counterfactual(self, x_factual, index, value):

        x_cf = torch.randn_like(x_factual)
        x_cf[:, index] = value

        return x_cf

    def log_prob(self, x):
        px = Independent(
            Normal(
                torch.zeros(7),
                torch.ones(7),
            ),
            1,
        )
        return px.log_prob(x)

    def _loss(self, loss):
        if loss in ["default", "forward"]:
            return "forward"
        else:
            raise NotImplementedError(f"Wrong loss {loss}")

    def _split_dataset(self, dataset_raw):
        datasets = []

        for i, split_s in enumerate(self.split):
            dataset = GermanDataset(
                root_dir=self.root, split=self.split_names[i], seed=self.k_fold
            )

            dataset.prepare_data()
            dataset.set_add_noise(self.add_noise)
            if i == 0:
                self.dataset = dataset
            datasets.append(dataset)

        return datasets

    def _get_dataset(self, num_samples, split_name):
        raise NotImplementedError

    def get_scaler(self, fit=True):

        scaler = self._get_scaler()
        self.scaler_transform = None
        if fit:
            x = self.get_features_train()
            scaler.fit(x, dims=self.dims_scaler)
            if self.scale in ["default", "std"]:
                self.scaler_transform = StandardTransform(
                    shift=x.mean(0), scale=x.std(0)
                )
                print("scaler_transform", self.scaler_transform)

        self.scaler = scaler

        return scaler

    def get_scaler_info(self):
        if self.scale in ["default", "std"]:
            return [("std", None)]
        else:
            raise NotImplementedError

    @property
    def dims_scaler(self):
        return (0,)

    def _get_dataset_raw(self):
        return None

    def _transform_dataset_pre_split(self, dataset_raw):
        return dataset_raw

    def post_process(self, x, inplace=False):
        if not inplace:
            x = x.clone()
        dims = self.dataset.binary_dims
        min_values = self.dataset.binary_min_values
        max_values = self.dataset.binary_max_values

        x[..., dims] = x[..., dims].floor().float()
        x[..., dims] = torch.clamp(x[..., dims], min=min_values, max=max_values)

        return x

    def feature_names(self, latex=False):
        return self.dataset.column_names


    def _plot_data(
        self,
        batch=None,
        title_elem_idx=None,
        batch_size=None,
        df=None,
        hue=None,
        **kwargs,
    ):

        title = ""
        return super()._plot_data(
            batch=batch,
            title_elem_idx=title_elem_idx,
            batch_size=batch_size,
            df=df,
            title=title,
            hue=hue,
            diag_plot="hist",
        )
