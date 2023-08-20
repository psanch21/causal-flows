from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import degree

from causal_nf.modules.gnn.nn.disjoint_dense import DisjointDense
from .pna_utils import AGGREGATORS, SCALERS

from causal_nf.utils.activations import get_act_fn


class DisjointPNAConv(MessagePassing):
    def __init__(
        self,
        m_channels: list,
        edge_dim: int,
        num_nodes: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor,
        act_name: Optional[str] = "relu",
        drop_rate: Optional[float] = 0.0,
        **kwargs
    ):
        """
        edge_dim is a one hot encoding of the index of the edge in the graph.
        I.e. edge_dim = # edges in our graph including self loops.
        """
        super(DisjointPNAConv, self).__init__(aggr=None, node_dim=0, **kwargs)

        assert len(m_channels) >= 2

        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]

        self.m_net_list = nn.ModuleList()
        self.activs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.m_layers = len(m_channels) - 1
        m_channels[0] = m_channels[0] * 2

        self.avg_deg: Dict[str, float] = {
            "lin": deg.mean().item(),
            "log": (deg + 1).log().mean().item(),
            "exp": deg.exp().mean().item(),
        }

        for i, (in_ch, out_ch) in enumerate(zip(m_channels[:-1], m_channels[1:])):
            m_net = DisjointDense(
                in_dimension=in_ch, out_dimension=out_ch, num_disjoint=edge_dim
            )
            self.m_net_list.append(m_net)
            act = get_act_fn(act_name)
            self.activs.append(act)

            dropout = nn.Dropout(drop_rate)
            self.dropouts.append(dropout)

        self.edge_dim = edge_dim

        self.in_update_net = (len(aggregators) * len(scalers)) * m_channels[
            -1
        ] + m_channels[0] // 2

        self.update_net = DisjointDense(
            in_dimension=self.in_update_net,
            out_dimension=m_channels[-1],
            num_disjoint=num_nodes,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m_net in self.m_net_list:
            m_net.reset_parameters()

        self.update_net.reset_parameters()

    def forward(
        self, x: Tensor, edge_index: Adj, edge_attr: Tensor, node_ids: Tensor
    ) -> Tensor:
        """
        edge_index = []
        edge_index.append([0,1])
        edge_index.append([2,2])
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)

        return self.update_net(out, one_hot_selector=node_ids)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        edge_attr: dimension self.edge_dim. In our case one-hot encoding
        x_i are N nodes being updated
        x_j is a neighbor of node x_i, could be itself if we have self-loops
        """

        x = torch.cat([x_i, x_j], dim=1)

        for i, (m_net, act, dout) in enumerate(
            zip(self.m_net_list, self.activs, self.dropouts)
        ):
            h = act(m_net(x, one_hot_selector=edge_attr))
            x = dout(h)
        return x

    def aggregate(
        self, inputs: Tensor, index: Tensor, dim_size: Optional[int] = None
    ) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1)

        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]

        return torch.cat(outs, dim=-1)
