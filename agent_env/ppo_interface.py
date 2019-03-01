from unityagents import UnityEnvironment
import numpy as np
from components import get_cfg_defaults
from agent import PPOAgent
import progressbar as pb
import torch.optim as optim


hyper_parameter = get_cfg_defaults().HYPER_PARAMETER.clone()
train_parameter = get_cfg_defaults().TRAIN_PARAMETER.clone()


env = UnityEnvironment(file_name='../Reacher_Env/multiple_agent/Reacher_Linux/Reacher.x86_64', no_graphics=True)
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


widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=train_parameter.EPISODES).start()

epsilon = hyper_parameter.EPSILON
beta = hyper_parameter.BETA


def train():
    global epsilon, beta
    agent = PPOAgent()
    optimizer = getattr(optim, train_parameter)(agent.policy.parameters(), lr=train_parameter.LR,
                                                momentum=train_parameter.MOMENTUM)
    mean_rewards = []
    for i in range(train_parameter.EPISODES):
        states, rewards, log_probs, actions, ent, dones = agent.collecct_trajectories(env, brain_name, tmax=500)
        # print('Episodes %d begin~~~~~' % i)
        # print('state info: ', len(states), states[0].shape)
        # print('rewards info: ', np.array(rewards).shape)
        # print('log_probs info: ', np.array(log_probs).shape)
        # print('actions info: ', len(actions), actions[0].shape)
        # print('ent info: ', np.array(ent).shape)
        # print()

        total_rewards = np.sum(rewards, axis=0)

        for _ in range(hyper_parameter.SURROGATE):
            loss = -agent.clip_surrogate(log_probs, states, actions, rewards, epsilon=epsilon, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss

        # the clipping parameter reduces as time goes on
        epsilon *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (i + 1) % 20 == 0:
            print("Episode: {0:d}, score: {1:f}".format(i + 1, mean_rewards))
            print(total_rewards)

        # update progress widget bar
        timer.update(i + 1)


if __name__ == '__main__':
    train()


