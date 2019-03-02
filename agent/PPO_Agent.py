import torch
from network import PPOPolicy
from components import get_cfg_defaults
import numpy as np

hyper_parameter = get_cfg_defaults().HYPER_PARAMETER.clone()
train_parameter = get_cfg_defaults().TRAIN_PARAMETER.clone()


class PPOAgent:
    def __init__(self, env):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy = PPOPolicy()
        self.policy.to(self.device)
        self.num_agents = hyper_parameter.AGENTS_NUM
        self.action_size = hyper_parameter.ACTION_SPACE
        self.tmax = hyper_parameter.TMAX
        self.env = env
        self.cum_rewards = 0.0
        self.rewards_std = 0.0

    def collecct_trajectories(self, brain_name, tmax=200):
        env = self.env
        # initialize returning lists and start the game!
        state_list = []
        reward_list = []
        log_probs = []
        prob_list = []
        action_list = []
        ent_list = []
        dones_list = []

        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # get the current state (for each agent)
        with torch.no_grad():
            for t in range(tmax):
                states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
                outs = self.policy(states_tensor)
                actions = outs['a'].cpu().detach().numpy()
                actions = np.clip(actions, -1.0, 1.0)
                lprob = outs['log_pi_a'].cpu().detach().numpy()
                prob = outs['prob'].cpu().detach().numpy()
                ent = outs['ent'].cpu().detach().numpy()
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = np.array(env_info.local_done)
                if dones.any():
                    break

                state_list.append(states)
                action_list.append(actions)
                reward_list.append(rewards)
                log_probs.append(lprob)
                ent_list.append(ent)
                dones_list.append(dones)
                prob_list.append(prob)
                states = next_states
        return state_list, np.array(reward_list), np.array(log_probs), action_list, \
               np.array(ent_list), dones_list, np.array(prob_list)

    def clip_surrogate(self, old_probs, states, actions, rewards, epsilon, beta):
        new_probs = torch.zeros(old_probs.shape[0], self.num_agents, self.action_size, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards_mean = torch.mean(rewards, dim=1, keepdim=True)
        rewards_std = torch.std(rewards, dim=1, keepdim=True, unbiased=True)
        rewards = (rewards - rewards_mean) / (rewards_std + 1e-10)
        rewards = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))
        rewards = rewards.unsqueeze(-1)

        old_probs = torch.tensor(old_probs, dtype=torch.float32, device=self.device)
        # convert states to policy (or probability)
        for i in range(len(states)):
            states_n = torch.tensor(states[i], dtype=torch.float32, device=self.device)
            action_n = torch.tensor(actions[i], dtype=torch.float32, device=self.device)
            outs = self.policy(states_n, action_n)
            new_probs[i] = outs['prob']
        new_probs = new_probs.to(self.device)
        ratio = torch.log(new_probs / (old_probs + 1.e-10))
        clip_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        ratio = torch.min(ratio, clip_ratio)
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                    (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))
        return torch.mean(ratio * rewards + beta * entropy)
