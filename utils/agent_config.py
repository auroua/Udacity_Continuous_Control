# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.HYPER_PARAMETER = CN()
# Replay Memory Size
_C.HYPER_PARAMETER.BUFFER_SIZE = int(1e5)
# Batch Size
_C.HYPER_PARAMETER.BATCH_SIZE = 64
# Soft Update of Target Parameters
_C.HYPER_PARAMETER.TAU = 1e-3
# Learning Rate
_C.HYPER_PARAMETER.LR = 5e-4
# MOMENTUM
_C.HYPER_PARAMETER.MOMENTUM = 0.9
# NetWork Update Frequency
_C.HYPER_PARAMETER.UPDATE_EVERY = 4
# Discount Factor
_C.HYPER_PARAMETER.GAMMA = 0.99
# Define the Loss Type. Support ('MSE', 'F1')
_C.HYPER_PARAMETER.LOSS_TYPE = 'F1'
# The Model Name Used to Approximate Q-Function  ('QNetwork',)
_C.HYPER_PARAMETER.MODEL_TYPE = 'QNetwork'
# The Training Agent Type
_C.HYPER_PARAMETER.AGENT_TYPE = 'DqnAgent'
# The Optimizer used for training (SGD, ADAM)
_C.HYPER_PARAMETER.OPTIMIZER = 'ADAM'
# Training algorithms
_C.HYPER_PARAMETER.INTERFACE = 'dqn'

_C.MODEL_PARAMETER = CN()
# Fully Connection Model Hidden Layer Parameter
_C.MODEL_PARAMETER.H1 = 512
_C.MODEL_PARAMETER.H2 = 512
_C.MODEL_PARAMETER.H3 = 512


def get_cfg_defaults():
    return _C.clone()
