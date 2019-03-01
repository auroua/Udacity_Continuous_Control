import torch
import torch.nn as nn
import torch.nn.functional as F
from components import get_cfg_defaults

hyper_parameter = get_cfg_defaults().HYPER_PARAMETER.clone()
model_parameter = get_cfg_defaults().MODEL_PARAMETER.clone()


class PPOPolicy(nn.Module):
    def __init__(self):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(hyper_parameter.STATE_SPACE, model_parameter.H1)
        self.fc2 = nn.Linear(model_parameter.H1, model_parameter.H1)
        self.fc3 = nn.Linear(model_parameter.H1, model_parameter.H2)
        self.fc4 = nn.Linear(model_parameter.H2, hyper_parameter.ACTION_SPACE)
        self.std = nn.Parameter(torch.zeros(hyper_parameter.AGENTS_NUM, hyper_parameter.ACTION_SPACE))

    def forward(self, x, action=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = F.dropout(F.relu(self.fc3(x)), p=0.5)
        mean = torch.tanh(self.fc4(x))

        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return {
            'a': action,
            'log_pi_a': log_prob,
            'mean': mean,
            'ent': entropy
        }

