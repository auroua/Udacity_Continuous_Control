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
# n-steps length
_C.HYPER_PARAMETER.ROLLOUT_LENGTH = 600
# MAX Single Episodes length
_C.HYPER_PARAMETER.TMAX = 10000
# Model Save Interval
_C.HYPER_PARAMETER.SAVE_INTERVAL = 5000
# Max Iteration Stpes
_C.HYPER_PARAMETER.MAX_STEPS = int(2e8)
# Model log interval
_C.HYPER_PARAMETER.LOG_INTERVAL = 500
# Use GAE
_C.HYPER_PARAMETER.USE_GAE = False
# GAE TAU
_C.HYPER_PARAMETER.GAE_TAU = 1.0


_C.MODEL_PARAMETER = CN()
# Fully Connection Model Hidden Layer Parameter
_C.MODEL_PARAMETER.H1 = 64
_C.MODEL_PARAMETER.H2 = 64


_C.TRAIN_PARAMETER = CN()
# Training episodes
_C.TRAIN_PARAMETER.EPISODES = 1000
# Learning Rate
_C.TRAIN_PARAMETER.LR =3e-4
# MOMENTUM
_C.TRAIN_PARAMETER.MOMENTUM = 0.9
# The Optimizer used for training (SGD, Adam, ADABOUND)
_C.TRAIN_PARAMETER.OPTIMIZER = 'Adam'
# Grad clip
_C.TRAIN_PARAMETER.Gradient_Clip = 5
# entropy weight
_C.TRAIN_PARAMETER.entropy_weight = 0.01
# value loss weight
_C.TRAIN_PARAMETER.value_loss_weight = 1.0


def get_ac_cfg_defaults():
    return _C.clone()
