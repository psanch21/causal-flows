import torch_geometric.nn as pygnn

from .base_gnn import BaseGNN
from .nn.disjoint_pna_conv import DisjointPNAConv


class MyDisjointPNAConv(DisjointPNAConv):
    def __init__(self, *args, **kwargs):
        super(MyDisjointPNAConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyDisjointPNAConv, self).forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            node_ids=batch.node_ids,
        )

        batch.x = x
        return batch


class DisjointPNA(BaseGNN):
    def __init__(
        self,
        *args,
        aggregators=[],
        scalers=[],
        deg=None,
        edge_dim=None,
        num_nodes=None,
        towers=1,
        **kwargs
    ):
        self.aggregators = aggregators
        self.scalers = scalers

        self.deg = deg

        self.edge_dim = edge_dim
        self.num_nodes = num_nodes

        self.towers = towers

        super(DisjointPNA, self).__init__(*args, **kwargs)

    @staticmethod
    def kwargs(cfg, preparator, input_size=None, output_size=None):
        my_dict = {}

        my_dict["aggregators"] = cfg.gnn.aggregators
        my_dict["scalers"] = cfg.gnn.scalers

        my_dict["deg"] = preparator.get_deg()

        my_dict["edge_dim"] = preparator.edge_attr_dim()
        my_dict["num_nodes"] = preparator.num_nodes

        my_dict["towers"] = cfg.gnn.towers

        my_dict.update(BaseGNN.kwargs(cfg, preparator, input_size, output_size))

        return my_dict

    def _gnn_layer(self, input_size, output_size, **kwargs):
        h_dim = (input_size + output_size) // 2
        if h_dim % 2 != 0:
            h_dim += 1
        m_channels = [input_size]

        for i in range(self.towers):
            m_channels.append(h_dim)
        m_channels.append(output_size)
        return MyDisjointPNAConv(
            m_channels=m_channels,
            edge_dim=self.edge_dim,
            num_nodes=self.num_nodes,
            aggregators=self.aggregators,
            scalers=self.scalers,
            deg=self.deg,
            act_name="relu",
            dropout=self.dropout,
            **kwargs
        )
