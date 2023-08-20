import torch.nn as nn
from causal_nf.utils.activations import get_act_fn


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        act_fn,
        device,
        has_bn=False,
        has_ln=False,
        use_act_out=False,
        dropout=0.0,
        drop_last=True,
    ):
        super(MLP, self).__init__()
        assert num_layers >= 0
        self.device = device
        self.output_size = output_size

        act_fn = get_act_fn(act_fn)

        layers = []
        for n in range(num_layers):

            input_size_i = None
            output_size_i = None
            blocks = []
            if n == 0:
                if num_layers == 1:
                    input_size_i = input_size
                    output_size_i = output_size
                    act = act_fn if use_act_out else nn.Identity().to(self.device)
                else:
                    input_size_i = input_size
                    output_size_i = hidden_size
                    act = act_fn
            elif n == (num_layers - 1):
                input_size_i = hidden_size
                output_size_i = output_size
                act = act_fn if use_act_out else nn.Identity().to(self.device)
            else:
                input_size_i = hidden_size
                output_size_i = hidden_size
                act = act_fn

            blocks.append(nn.Linear(input_size_i, output_size_i, device=self.device))
            if has_bn:
                blocks.append(nn.BatchNorm1d(output_size_i, device=self.device))
            elif has_ln:
                blocks.append(nn.LayerNorm(output_size_i, device=self.device))
            blocks.append(act)

            if dropout > 0.0 and (n < (num_layers - 1) or drop_last):
                drop = nn.Dropout(dropout).to(device)
                blocks.append(drop)
            layers.append(nn.Sequential(*blocks).to(self.device))

        if num_layers == 0:
            layers = [nn.Identity().to(device)]
            if input_size != output_size:
                raise ValueError(
                    "input_size != output_size [{}, {}]".format(input_size, output_size)
                )

        self.fc_layers = nn.ModuleList(layers)

    @staticmethod
    def kwargs(cfg, preparator):
        my_dict = {}
        my_dict["input_size"] = preparator.x_dim()
        my_dict["hidden_size"] = cfg.model.dim_inner
        my_dict["output_size"] = cfg.model.label_dim()
        my_dict["num_layers"] = cfg.model.num_layers
        my_dict["act_fn"] = cfg.model.act
        my_dict["dropout"] = cfg.model.dropout

        return my_dict

    def forward(self, x):
        for fc in self.fc_layers:
            x = fc(x)
        return x
