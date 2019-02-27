from unityagents import UnityEnvironment
import numpy as np
from components import get_cfg_defaults

hyper_parameter = get_cfg_defaults().HYPER_PARAMETER.clone()

env = UnityEnvironment(file_name='../Reacher_Env/single_agent/Reacher_Linux/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('Each observes a state with length : %d' % state_size)


def step_reacher(policy):
    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    while True:
        pass
