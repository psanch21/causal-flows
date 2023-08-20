from abc import abstractmethod

import torch.nn as nn

from causal_nf.modules.mlp import MLP
from causal_nf.utils.activations import get_act_fn

import torch_geometric.nn as pygnn
from .base_gnn import BaseGNN


class MyGATConv(pygnn.GATConv):
    def __init__(self, *args, **kwargs):
        super(MyGATConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyGATConv, self).forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            return_attention_weights=None,
        )

        batch.x = x
        return batch


class GAT(BaseGNN):
    def __init__(self, *args, heads=1, edge_dim=None, **kwargs):
        self.heads = heads
        self.edge_dim = edge_dim
        super(GAT, self).__init__(*args, **kwargs)

    @staticmethod
    def kwargs(cfg, preparator, input_size=None, output_size=None):
        my_dict = {}
        my_dict["heads"] = cfg.gnn.heads
        my_dict["edge_dim"] = preparator.edge_attr_dim()
        my_dict.update(BaseGNN.kwargs(cfg, preparator, input_size, output_size))

        return my_dict

    def _gnn_layer(self, input_size, output_size, **kwargs):
        assert output_size % self.heads == 0
        out_channels = output_size // self.heads
        return MyGATConv(
            in_channels=input_size,
            out_channels=out_channels,
            heads=self.heads,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            edge_dim=self.edge_dim,
            fill_value="mean",
            bias=True,
            **kwargs
        )
