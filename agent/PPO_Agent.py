import torch
from network import PPOPolicy
from components import get_cfg_defaults
import numpy as np

hyper_parameter = get_cfg_defaults().HYPER_PARAMETER.clone()


class PPOAgent:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy = PPOPolicy()
        self.policy.to(self.device)

    def collecct_trajectories(self, env, brain_name, tmax=200):
        # initialize returning lists and start the game!
        state_list = []
        reward_list = []
        log_probs = []
        action_list = []
        ent_list = []
        dones_list = []

        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # get the current state (for each agent)
        for t in range(tmax):
            states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            outs = self.policy(states_tensor)

            actions = outs['a'].cpu().detach().numpy()
            lprob = outs['log_pi_a'].cpu().detach().numpy()
            ent = outs['ent'].cpu().detach().numpy()
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = np.array(env_info.local_done)

            state_list.append(states_tensor)
            action_list.append(actions)
            reward_list.append(rewards)
            log_probs.append(lprob)
            ent_list.append(ent)
            dones_list.append(dones)
            if dones.any():
                print('episode done')
                break
            states = next_states
        return state_list, reward_list, log_probs, action_list, ent_list, dones_list


    def surrogate(self, policy, old_probs, states, actions, rewards, discount=0.995, beta=0.01):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))
        rewards_mean = torch.mean(rewards, dim=1, keepdim=True)
        rewards_std = torch.std(rewards, dim=1, keepdim=True, unbiased=True)
        rewards = (rewards - rewards_mean) / (rewards_std + 1e-10)

        actions = torch.tensor(actions, dtype=torch.int8, device=device)

        old_probs = torch.tensor(old_probs, dtype=torch.float32, device=device)
        # convert states to policy (or probability)
        new_probs = pong_utils.states_to_prob(policy, states)
        new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)

        ratio = new_probs / (old_probs + 1.e-10)

        clip_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        ratio = torch.min(ratio, clip_ratio)
        # include a regularization term
        # this steers new_policy towards 0.5
        # prevents policy to become exactly 0 or 1 helps exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                    (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        return torch.mean(ratio * rewards + beta * entropy)

    def clip_surrogate(self, old_probs, states, actions, rewards, epsilon, beta):
        new_probs = []
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards_mean = torch.mean(rewards, dim=1, keepdim=True)
        rewards_std = torch.std(rewards, dim=1, keepdim=True, unbiased=True)
        rewards = (rewards - rewards_mean) / (rewards_std + 1e-10)
        rewards = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

        old_probs = torch.tensor(old_probs, dtype=torch.float32, device=self.device)
        # convert states to policy (or probability)
        for i in range(len(states)):
            outs = self.policy(states[i])

        new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)
        ratio = new_probs / (old_probs + 1.e-10)
        clip_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        ratio = torch.min(ratio, clip_ratio)
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                    (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))
        return torch.mean(ratio * rewards + beta * entropy)
