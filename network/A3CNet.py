import torch
import torch.nn as nn
import torch.nn.functional as F


class A3CNet(nn.Module):
    def __init__(self):
        super(A3CNet, self).__init__()

    def forward(self, x):
        ...
