import torch.nn as nn


def init_xavier(act_fn="relu", **kwargs):
    def _init_xavier(m):
        """Performs weight initialization."""
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data = nn.init.xavier_uniform_(
                m.weight.data, gain=nn.init.calculate_gain(act_fn, **kwargs)
            )
            if m.bias is not None:
                m.bias.data.zero_()

    return _init_xavier
