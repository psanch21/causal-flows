from causal_nf.datasets.scm_dataset import SCMDataset
from causal_nf.distributions.scm import SCM
from causal_nf.preparators.scm._base_distributions import pu_dict
from causal_nf.preparators.tabular_preparator import TabularPreparator
from causal_nf.sem_equations import sem_dict
from causal_nf.transforms import CausalEquations
from causal_nf.utils.io import dict_to_cn

from causal_nf.utils.scalers import StandardTransform

import numpy as np

import networkx as nx
import torch.utils.data as tdata
import torch_geometric.data as pygdata

from .batch_generator import BatchGenerator
from torch_geometric.utils import degree


import torch


class SCMPreparator(TabularPreparator):
    def __init__(
        self,
        name,
        num_samples,
        sem_name,
        base_version,
        type="torch",
        use_edge_attr=False,
        **kwargs,
    ):

        self._num_samples = num_samples
        self.sem_name = sem_name
        self.dataset = None
        self.type = type
        self.use_edge_attr = use_edge_attr
        sem_fn = sem_dict[name](sem_name=sem_name)

        self.adjacency = sem_fn.adjacency

        sem = CausalEquations(
            functions=sem_fn.functions, inverses=sem_fn.inverses, derivatives=None
        )
        self.num_nodes = len(sem_fn.functions)
        base_distribution = pu_dict[self.num_nodes](base_version)

        self.base_version = base_version
        self.scm = SCM(base_distribution=base_distribution, transform=sem)

        self.intervention_index_list = sem_fn.intervention_index_list()

        super().__init__(name=name, task="modeling", **kwargs)

    @classmethod
    def params(cls, dataset):
        if isinstance(dataset, dict):
            dataset = dict_to_cn(dataset)

        my_dict = {
            "name": dataset.name,
            "num_samples": dataset.num_samples,
            "sem_name": dataset.sem_name,
            "base_version": dataset.base_version,
            "type": dataset.type,
            "use_edge_attr": dataset.use_edge_attr,
        }

        my_dict.update(TabularPreparator.params(dataset))

        return my_dict

    @classmethod
    def loader(cls, dataset):
        my_dict = SCMPreparator.params(dataset)

        return cls(**my_dict)

    def _x_dim(self):
        if self.type == "torch":
            return self.num_nodes
        elif self.type == "pyg":
            return 1

    def edge_attr_dim(self):
        if self.dataset.use_edge_attr:
            return self.dataset.edge_attr.shape[-1]
        else:
            return None

    def feature_names(self, latex=False):
        if self.type == "torch":
            x_dim = self.x_dim()
        else:
            x_dim = self.num_nodes
        if latex:
            return [f"$x_{{{i + 1}}}$" for i in range(x_dim)]
        else:
            return [f"x_{i + 1}" for i in range(x_dim)]

    def get_deg(self):

        loader = self.get_dataloader_train(batch_size=1)

        max_degree = 0
        i = 0
        for data in loader:
            i += 1
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
            # Compute the in-degree histogram tensor
            if i == 100:
                break
        deg_histogram = torch.zeros(max_degree + 1, dtype=torch.long)
        i = 0
        for data in loader:
            i += 1
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg_histogram += torch.bincount(d, minlength=deg_histogram.numel())
            if i == 100:
                break
        return deg_histogram.float()

    def get_batch_generator(self):
        if self.type == "pyg":
            return BatchGenerator(
                node_dim=1,
                num_nodes=self.num_nodes,
                edge_index=self.dataset.edge_index,
                device="cpu",
                edge_attr=self.dataset.edge_attr,
                node_ids=self.dataset.node_ids,
            )

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

    def get_features_train(self):
        loader = self.get_dataloader_train(batch_size=self.num_samples())
        batch = next(iter(loader))
        if self.type == "torch":
            return batch[0]
        elif self.type == "pyg":
            return batch.x.reshape(batch.num_graphs, -1)
        else:
            raise NotImplementedError(f"Type {self.type} not implemented")

    def _data_loader(self, dataset, batch_size, shuffle, num_workers=0):
        if self.type == "torch":
            return tdata.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=False,
            )
        elif self.type == "pyg":
            return pygdata.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=False,
            )

    def get_intervention_list_2(self):
        x = self.get_features_train()

        x_mean = x.mean(0)
        x_std = x.std(0)
        int_list = []
        for i in self.intervention_index_list:
            x_mean_i = x_mean[i].item()
            x_std_i = x_std[i].item()
            values_i = []
            values_i.append({"name": "0std", "value": x_mean_i})
            values_i.append({"name": "-1std", "value": x_mean_i - x_std_i})
            values_i.append({"name": "-2std", "value": x_mean_i - 2 * x_std_i})
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
        self.scm.intervene(index, value)
        x_int = self.scm.sample(shape)
        self.scm.stop_intervening(index)
        return x_int

    def compute_ate(self, index, a, b, num_samples=10000):
        ate = self.scm.compute_ate(index, a, b, num_samples)
        return ate

    def compute_counterfactual(self, x_factual, index, value):
        u = self.scm.transform.inv(x_factual)
        self.scm.intervene(index, value)
        x_cf = self.scm.transform(u)
        self.scm.stop_intervening(index)
        return x_cf

    def log_prob(self, x):
        return self.scm.log_prob(x)

    def _loss(self, loss):
        if loss in ["default", "forward"]:
            return "forward"
        else:
            raise NotImplementedError(f"Wrong loss {loss}")

    def _split_dataset(self, dataset_raw):
        datasets = []

        for i, split_s in enumerate(self.split):
            num_samples = int(self._num_samples * split_s)
            if self.k_fold >= 0:
                seed = self.k_fold + i * 100
            else:
                seed = None

            dataset = SCMDataset(
                root_dir=self.root,
                num_samples=num_samples,
                scm=self.scm,
                name=self.name,
                sem_name=self.sem_name,
                type=self.type,
                use_edge_attr=self.use_edge_attr,
                seed=seed,
            )

            dataset.prepare_data()
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
        )
