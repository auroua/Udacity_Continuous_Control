import torch
import torch.nn as nn

LAYER1_IN_FEATURE = 128
LAYER2_IN_FEATURE = 256


class PPOPolicy(nn.Module):
    def __init__(self):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(33, 128)

    def forward(self, x):
        ...
