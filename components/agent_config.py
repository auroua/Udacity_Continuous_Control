# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()


_C.HYPER_PARAMETER = CN()
# Replay Memory Size
_C.HYPER_PARAMETER.BUFFER_SIZE = int(1e5)
# The EPSILON Value
_C.HYPER_PARAMETER.EPSILON = 0.2
# Soft Update of Target Parameters
_C.HYPER_PARAMETER.TAU = 1e-3
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
_C.HYPER_PARAMETER.ROLLOUT_LENGTH = 128
# Surrogate length
_C.HYPER_PARAMETER.SURROGATE = 4
# MAX Single Episodes length
_C.HYPER_PARAMETER.TMAX = 100
# Suggort clip value
_C.HYPER_PARAMETER.CLIP_VAL = 0.2


_C.MODEL_PARAMETER = CN()
# Fully Connection Model Hidden Layer Parameter
_C.MODEL_PARAMETER.H1 = 128
_C.MODEL_PARAMETER.H2 = 256


_C.TRAIN_PARAMETER = CN()
# Training episodes
_C.TRAIN_PARAMETER.EPISODES = 1000
# Batch Size
_C.TRAIN_PARAMETER.BATCH_SIZE = 64
# Learning Rate
_C.TRAIN_PARAMETER.LR = 3e-4
# MOMENTUM
_C.TRAIN_PARAMETER.MOMENTUM = 0.9
# The Optimizer used for training (SGD, Adam)
_C.TRAIN_PARAMETER.OPTIMIZER = 'Adam'
# The Proximal Policy Optimization SGD STEP
_C.TRAIN_PARAMETER.SGD_EPOCH = 4
# The AGENT AND ENVIRONMENT USED TO LEARN
_C.TRAIN_PARAMETER.AE_TYPE = 'ppo_reacher_interaction'
_C.TRAIN_PARAMETER.AGENT_TYPE = 'PPOAgent'
# Grad clip
_C.TRAIN_PARAMETER.Gradient_Clip = 0.5


def get_cfg_defaults():
    return _C.clone()
