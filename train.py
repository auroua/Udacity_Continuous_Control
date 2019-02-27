import agent_env
from components import get_cfg_defaults
import torch.optim as optim
import agent


hyper_parameter = get_cfg_defaults().HYPER_PARAMETER.clone()
train_parameter = get_cfg_defaults().TRAIN_PARAMETER.clone()

if __name__ == '__main__':
    agent = getattr(agent, train_parameter.AGENT_TYPE)
    optimizer = getattr(optim, train_parameter.OPTIMIZER)(parameters=agent.policy.parameters(),
                                                          lr=train_parameter.LR,
                                                          momentum=train_parameter.MOMENTUM)
    getattr(agent_env, train_parameter.AE_TYPE)(agent, optimizer)
