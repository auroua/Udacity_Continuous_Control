# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()


_C.HYPER_PARAMETER = CN()
# Replay Memory Size
_C.HYPER_PARAMETER.BUFFER_SIZE = int(1e5)
# The EPSILON Value
_C.HYPER_PARAMETER.EPSILON = 0.1
# Soft Update of Target Parameters
_C.HYPER_PARAMETER.TAU = 1e-3
# The beta parameter used to balance rewards and entropy
_C.HYPER_PARAMETER.BETA = 0.01
# NetWork Update Frequency
_C.HYPER_PARAMETER.UPDATE_EVERY = 4
# Discount Factor
_C.HYPER_PARAMETER.GAMMA = 0.99
# Define the Loss Type. Support ('MSE', 'F1')
_C.HYPER_PARAMETER.LOSS_TYPE = 'F1'
# Action Space Size
_C.HYPER_PARAMETER.ACTION_SPACE = 4
# Environment State Space Size
_C.HYPER_PARAMETER.STATE_SPACE = 33
_C.HYPER_PARAMETER.ROLLOUT_LEN = 128

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
_C.TRAIN_PARAMETER.LR = 5e-4
# MOMENTUM
_C.TRAIN_PARAMETER.MOMENTUM = 0.9
# The Optimizer used for training (SGD, ADAM)
_C.TRAIN_PARAMETER.OPTIMIZER = 'ADAM'
# The Proximal Policy Optimization SGD STEP
_C.TRAIN_PARAMETER.SGD_EPOCH = 4
# The AGENT AND ENVIRONMENT USED TO LEARN
_C.TRAIN_PARAMETER.AE_TYPE = 'ppo_reacher_interaction'
_C.TRAIN_PARAMETER.AGENT_TYPE = 'PPOAgent'


def get_cfg_defaults():
    return _C.clone()
