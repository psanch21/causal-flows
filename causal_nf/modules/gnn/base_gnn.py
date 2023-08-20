from abc import abstractmethod

import torch.nn as nn

from causal_nf.modules.mlp import MLP
from causal_nf.utils.activations import get_act_fn
from .node_wrapper import NodeWrapper


class BaseGNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        act_fn,
        dropout,
        has_bn,
        stage_type,
        num_layers_pre,
        num_layers_gnn,
        num_layers_post,
        device,
    ):
        super(BaseGNN, self).__init__()
        self.device = device
        self.has_pre = num_layers_pre > 0
        self.has_gnn = num_layers_gnn > 0
        self.has_post = num_layers_post > 0

        self.num_layers_pre = num_layers_pre
        self.num_layers_gnn = num_layers_gnn
        self.num_layers_post = num_layers_post

        self.output_size = output_size

        self.act_fn = get_act_fn(act_fn)
        self.has_bn = has_bn
        self.stage_type = stage_type
        self.lin_skipsum = None

        self.dropout = dropout

        if self.has_pre:
            input_size_pre = input_size
            output_size_pre = hidden_size
        else:
            input_size_pre = input_size
            output_size_pre = input_size

        self.pre_nn = MLP(
            input_size_pre,
            hidden_size=hidden_size,
            output_size=output_size_pre,
            num_layers=num_layers_pre,
            act_fn=act_fn,
            has_bn=has_bn,
            use_act_out=self.has_gnn or self.has_post,
            dropout=dropout,
            device=self.device,
        )

        if num_layers_post > 0:
            output_size_gnn = hidden_size
        else:
            output_size_gnn = output_size

        self.gnn = self._build_gnn(
            input_size=self.pre_nn.output_size,
            hidden_size=hidden_size,
            output_size=output_size_gnn,
            act_fn=act_fn,
            dropout=dropout,
            num_layers=num_layers_gnn,
            act_last=num_layers_post > 0,
        )

        if num_layers_gnn > 0:
            input_size_post = hidden_size
        else:
            input_size_post = self.pre_nn.output_size

        self.post_nn = MLP(
            input_size=input_size_post,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers_post,
            act_fn=act_fn,
            has_bn=has_bn,
            dropout=dropout,
            drop_last=True,
            device=self.device,
        )

    @staticmethod
    def kwargs(cfg, preparator, input_size=None, output_size=None):
        my_dict = {}
        if input_size is None:
            input_size = preparator.x_dim()

        my_dict["input_size"] = input_size
        my_dict["hidden_size"] = cfg.gnn.dim_inner
        if output_size is None:
            output_size = cfg.model.latent_dim

        my_dict["output_size"] = output_size

        my_dict["act_fn"] = cfg.model.act
        my_dict["dropout"] = cfg.model.dropout
        my_dict["has_bn"] = cfg.model.has_bn

        my_dict["stage_type"] = cfg.gnn.stage_type

        my_dict["num_layers_pre"] = cfg.gnn.num_layers_pre
        my_dict["num_layers_gnn"] = cfg.gnn.num_layers
        my_dict["num_layers_post"] = cfg.gnn.num_layers_post
        my_dict["device"] = cfg.device

        return my_dict

    def forward(self, batch, inplace=False, **kwargs):
        if not inplace:
            batch = batch.clone()
        batch = self.forward_pre(batch)
        batch = self.forward_gnn(batch, **kwargs)
        batch = self.forward_post(batch)

        logits = batch.x
        return logits

    def forward_pre(self, batch):
        x = self.pre_nn(batch.x)

        batch.x = x

        return batch

    @abstractmethod
    def _gnn_layer(self, input_size, output_size, **kwargs):
        pass

    def _build_gnn(
        self,
        input_size,
        hidden_size,
        output_size,
        act_fn,
        dropout,
        num_layers,
        act_last,
    ):

        assert num_layers >= 0
        act_fn = NodeWrapper(get_act_fn(act_fn))

        layers = []

        if self.stage_type == "skipsum":
            linears = []
        for n in range(num_layers):

            input_size_i = None
            output_size_i = None
            act_i = None

            if n == 0:
                if num_layers == 1 and not act_last:
                    input_size_i = input_size
                    output_size_i = output_size
                    act_i = NodeWrapper(nn.Identity())
                else:
                    input_size_i = input_size
                    output_size_i = hidden_size
                    act_i = act_fn
            elif n == (num_layers - 1) and not act_last:
                input_size_i = hidden_size
                output_size_i = hidden_size
                act_i = NodeWrapper(nn.Identity())
            else:
                input_size_i = hidden_size
                output_size_i = hidden_size
                act_i = act_fn

            if self.stage_type == "skipsum":
                tmp = [nn.Linear(input_size_i, output_size)]
                if dropout > 0.0:
                    tmp.append(nn.Dropout(dropout))
                linears.append(nn.Sequential(*tmp))

            l = self._gnn_layer(
                input_size=input_size_i,
                output_size=output_size_i,
                flow="target_to_source",
            )
            layers_i = [l]
            if self.has_bn:
                layers_i.append(NodeWrapper(nn.BatchNorm1d(output_size_i)))

            layers_i.append(act_i)
            if dropout > 0.0:
                layers_i.append(NodeWrapper(nn.Dropout(dropout)))

            layers.append(nn.Sequential(*layers_i))

        if num_layers == 0:
            layers = [nn.Identity()]

        if self.stage_type == "skipsum":
            self.lin_skipsum = nn.ModuleList(linears)

        return nn.ModuleList(layers)

    def forward_gnn(self, batch, **kwargs):
        out = 0.0
        if self.num_layers_gnn == 0:
            return batch
        for i, l in enumerate(self.gnn):
            if self.stage_type == "skipsum":
                out += self.lin_skipsum[i](batch.x)

            batch = l(batch, **kwargs)

        if self.stage_type == "skipsum":
            out += batch.x
            batch.x = out
        return batch

    def forward_post(self, batch):
        x = self.post_nn(batch.x)
        batch.x = x
        return batch
