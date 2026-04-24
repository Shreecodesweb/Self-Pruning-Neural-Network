import torch.nn as nn
from prunable_layer import PrunableLinear

class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.ReLU(),
            PrunableLinear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
