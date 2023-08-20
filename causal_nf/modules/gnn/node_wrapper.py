import torch.nn as nn


class NodeWrapper(nn.Module):
    def __init__(self, layer):
        super(NodeWrapper, self).__init__()

        self.layer = layer

    def forward(self, batch, **kwargs):
        batch.x = self.layer(batch.x, *kwargs)
        return batch
