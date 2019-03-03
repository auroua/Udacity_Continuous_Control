# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.HYPER_PARAMETER = CN()
# The EPSILON Value
_C.HYPER_PARAMETER.EPSILON = 0.2
# The beta parameter used to balance rewards and entropy
_C.HYPER_PARAMETER.BETA = 0.01
# NetWork Update Frequency
_C.HYPER_PARAMETER.UPDATE_EVERY = 4
# Discount Factor
_C.HYPER_PARAMETER.GAMMA = 0.99
# The Agent Number
_C.HYPER_PARAMETER.AGENTS_NUM = 20
# Action Space Size
_C.HYPER_PARAMETER.ACTION_SPACE = 4
# Environment State Space Size
_C.HYPER_PARAMETER.STATE_SPACE = 33
# Surrogate length
_C.HYPER_PARAMETER.SURROGATE = 10
# MAX Single Episodes length
_C.HYPER_PARAMETER.TMAX = 10000
# Suggort clip value
_C.HYPER_PARAMETER.CLIP_VAL = 0.2


_C.MODEL_PARAMETER = CN()
# Fully Connection Model Hidden Layer Parameter
_C.MODEL_PARAMETER.H1 = 64
_C.MODEL_PARAMETER.H2 = 64


_C.TRAIN_PARAMETER = CN()
# Training episodes
_C.TRAIN_PARAMETER.EPISODES = 1000
# Learning Rate
_C.TRAIN_PARAMETER.LR =1e-4
# MOMENTUM
_C.TRAIN_PARAMETER.MOMENTUM = 0.9
# The Optimizer used for training (SGD, Adam, ADABOUND)
_C.TRAIN_PARAMETER.OPTIMIZER = 'Adam'
# Grad clip
_C.TRAIN_PARAMETER.Gradient_Clip = 5


def get_cfg_defaults():
    return _C.clone()
