import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def to_np(t):
    return t.cpu().detach().numpy()


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]
