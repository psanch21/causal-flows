import torch_geometric.nn as pygnn
from .base_gnn import BaseGNN
import torch.nn as nn


class MyGINConv(pygnn.GINConv):
    def __init__(self, *args, **kwargs):
        super(MyGINConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyGINConv, self).forward(x=batch.x, edge_index=batch.edge_index)

        batch.x = x
        return batch


class GIN(BaseGNN):
    def __init__(self, *args, eps=0.0, train_eps=False, **kwargs):
        self.eps = eps
        self.train_eps = train_eps
        super(GIN, self).__init__(*args, **kwargs)

    @staticmethod
    def kwargs(cfg, preparator, input_size=None, output_size=None):
        my_dict = {}
        my_dict["eps"] = cfg.gnn.eps
        my_dict["train_eps"] = cfg.gnn.train_eps
        my_dict.update(BaseGNN.kwargs(cfg, preparator, input_size, output_size))

        return my_dict

    def _gnn_layer(self, input_size, output_size, **kwargs):
        layers = [nn.Linear(input_size, output_size)]
        if self.has_bn:
            layers.append(nn.BatchNorm1d(output_size))
        layers.append(self.act_fn)
        layers.append(nn.Linear(output_size, output_size))

        net = nn.Sequential(*layers)
        return MyGINConv(
            nn=net, eps=self.eps, train_eps=self.train_eps, aggr="add", **kwargs
        )
