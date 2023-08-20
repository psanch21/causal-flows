import torch_geometric.nn as pygnn

from .base_gnn import BaseGNN


class MyGCNConv(pygnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(MyGCNConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyGCNConv, self).forward(
            x=batch.x, edge_index=batch.edge_index, edge_weight=None
        )

        batch.x = x
        return batch


class GCN(BaseGNN):
    def __init__(self, *args, **kwargs):
        super(GCN, self).__init__(*args, **kwargs)

    @staticmethod
    def kwargs(cfg, preparator):
        my_dict = {}
        my_dict.update(BaseGNN.kwargs(cfg, preparator))

        return my_dict

    def _gnn_layer(self, input_size, output_size, **kwargs):
        return MyGCNConv(
            in_channels=input_size,
            out_channels=output_size,
            improved=False,
            cached=False,
            add_self_loops=True,
            normalize=True,
            bias=True,
            **kwargs
        )
