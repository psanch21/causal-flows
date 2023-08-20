import torch.nn as nn
import torch_geometric.nn as pygnn

from .base_gnn import BaseGNN


class MyGINEConv(pygnn.GINEConv):
    def __init__(self, *args, **kwargs):
        super(MyGINEConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyGINEConv, self).forward(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
        )

        batch.x = x
        return batch


class GINE(BaseGNN):
    def __init__(self, *args, eps=0.0, train_eps=False, edge_dim=None, **kwargs):
        self.eps = eps
        self.train_eps = train_eps
        self.edge_dim = edge_dim
        super(GINE, self).__init__(*args, **kwargs)

    @staticmethod
    def kwargs(cfg, preparator):
        my_dict = {}
        my_dict["eps"] = cfg.gnn.eps
        my_dict["train_eps"] = cfg.gnn.train_eps
        my_dict["edge_dim"] = preparator.edge_attr_dim()
        my_dict.update(BaseGNN.kwargs(cfg, preparator))

        return my_dict

    def _gnn_layer(self, input_size, output_size):
        layers = [nn.Linear(input_size, output_size)]
        if self.has_bn:
            layers.append(nn.BatchNorm1d(output_size))
        layers.append(self.act_fn)
        layers.append(nn.Linear(output_size, output_size))

        net = nn.Sequential(*layers)
        return MyGINEConv(
            nn=net,
            eps=self.eps,
            train_eps=self.train_eps,
            edge_dim=self.edge_dim,
            aggr="add",
        )
