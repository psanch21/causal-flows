import torch_geometric.nn as pygnn

from .base_gnn import BaseGNN


class MyPNAConv(pygnn.PNAConv):
    def __init__(self, *args, **kwargs):
        super(MyPNAConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyPNAConv, self).forward(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
        )

        batch.x = x
        return batch


class PNA(BaseGNN):
    def __init__(
        self,
        *args,
        aggregators=[],
        scalers=[],
        deg=None,
        edge_dim=None,
        towers=1,
        pre_layers=1,
        post_layers=1,
        **kwargs
    ):
        self.aggregators = aggregators
        self.scalers = scalers

        self.deg = deg

        self.edge_dim = edge_dim

        self.towers = towers
        self.pre_layers = pre_layers
        self.post_layers = post_layers

        super(PNA, self).__init__(*args, **kwargs)

    @staticmethod
    def kwargs(cfg, preparator, input_size=None, output_size=None):
        my_dict = {}

        my_dict["aggregators"] = cfg.gnn.aggregators
        my_dict["scalers"] = cfg.gnn.scalers

        my_dict["deg"] = preparator.get_deg()

        my_dict["edge_dim"] = preparator.edge_attr_dim()

        my_dict["towers"] = cfg.gnn.towers
        my_dict["pre_layers"] = cfg.gnn.pre_layers
        my_dict["post_layers"] = cfg.gnn.post_layers

        my_dict.update(BaseGNN.kwargs(cfg, preparator, input_size, output_size))

        return my_dict

    def _gnn_layer(self, input_size, output_size, **kwargs):
        return MyPNAConv(
            in_channels=input_size,
            out_channels=output_size,
            aggregators=self.aggregators,
            scalers=self.scalers,
            deg=self.deg,
            edge_dim=self.edge_dim,
            towers=self.towers,
            pre_layers=self.pre_layers,
            post_layers=self.post_layers,
        )
