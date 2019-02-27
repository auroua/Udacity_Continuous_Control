from unityagents import UnityEnvironment
import numpy as np
from components import get_cfg_defaults
import progressbar
from agent import PPOAgent


hyper_parameter = get_cfg_defaults().HYPER_PARAMETER.clone()
train_parameter = get_cfg_defaults().TRAIN_PARAMETER.clone()

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


def ppo_reacher_interaction(agent, optimizer):
    epsilon = hyper_parameter.EPSILON
    beta = hyper_parameter.BETA
    for e in range(train_parameter.EPISODES):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:
            actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
            actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step

            # gradient ascent step
            for _ in range(train_parameter.SGD_EPOCH):
                L = -agent.clipped_surrogate(old_prob, states, actions, rewards, beta=beta)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
                del L
            if np.any(dones):  # exit loop if episode finished
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

        # the clipping parameter reduces as time goes on
        epsilon *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e + 1) % 20 == 0:
            print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
            print(total_rewards)

        # update progress widget bar
        timer.update(e + 1)

    env.close()


if __name__ == '__main__':
    pass
    # env_agent_loop()



    # discount_rate = .99
    # epsilon = 0.1
    # beta = .01
    # tmax = 320
    # SGD_epoch = 4
    #
    # # keep track of progress
    # mean_rewards = []
    #
    # for e in range(train_parameter.EPISODES):
    #     # collect trajectories
    #     old_probs, states, actions, rewards = \
    #         pong_utils.collect_trajectories(envs, policy, tmax=tmax)
    #
    #     total_rewards = np.sum(rewards, axis=0)
    #

    #
    #     # the clipping parameter reduces as time goes on
    #     epsilon *= .999
    #
    #     # the regulation term also reduces
    #     # this reduces exploration in later runs
    #     beta *= .995
    #
    #     # get the average reward of the parallel environments
    #     mean_rewards.append(np.mean(total_rewards))
    #
    #     # display some progress every 20 iterations
    #     if (e + 1) % 20 == 0:
    #         print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
    #         print(total_rewards)
    #
    #     # update progress widget bar
    #     timer.update(e + 1)
    #
    # timer.finish()