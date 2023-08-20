from torch_geometric.data import Batch, Data
import torch

from torch_geometric.utils import (
    to_dense_adj,
    dense_to_sparse,
    add_remaining_self_loops,
)


class BatchGenerator(object):
    def __init__(self, node_dim, num_nodes, edge_index, device="cpu", **kwargs):
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.has_self_loops = (edge_index[0] == edge_index[1]).any()

        self.adj = to_dense_adj(edge_index)[0]
        self.device = device
        self.extra_attr = {}
        for key, value in kwargs.items():
            self.extra_attr[key] = value

    def __call__(self, num_graphs):
        data_list = []
        for i in range(num_graphs):
            attr_dict = {}
            attr_dict["x"] = torch.zeros(
                [self.num_nodes, self.node_dim], device=self.device
            )
            attr_dict["edge_index"] = self.edge_index
            for key, value in self.extra_attr.items():
                attr_dict[key] = value

            data = Data(**attr_dict)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        return batch

    def edge_index_intervene(self, index):

        edge_index = self.edge_index.clone()
        mask_1 = edge_index[0, :] != index
        mask_2 = edge_index[0, :] == edge_index[1, :]
        mask = torch.bitwise_or(mask_1, mask_2)
        edge_index = edge_index[:, mask]
        return edge_index, mask

    def intervene(self, batch, index, value):
        edge_index_i, mask = self.edge_index_intervene(index)
        data_list = batch.to_data_list()
        data_out = []
        for data in data_list:
            data = data.clone()
            data.x[index, :] = value
            data.edge_index = edge_index_i
            for key, value_ in self.extra_attr.items():
                if "edge" in key:
                    value_ = value_[mask, ...]
                setattr(data, key, value_)
            data_out.append(data)

        batch_i = Batch.from_data_list(data_out)

        return batch_i
