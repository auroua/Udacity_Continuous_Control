import torch
import torch.nn.functional as F


class A3CAgent(torch.nn.Module):
    def __init__(self):
        super(A3CAgent, self).__init__()

    def forward(self, x):
        pass