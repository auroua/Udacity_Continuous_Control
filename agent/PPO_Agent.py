import torch
from network import PPOPolicy
from components import get_cfg_defaults

hyper_parameter = get_cfg_defaults().HYPER_PARAMETER.clone()


class PPOAgent:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy = PPOPolicy()

    def step(self, states):


        states = torch.Tensor(states, dtype=torch.float32, device=self.device)
        states = states.view(1, -1)
        actions = self.policy(states)
        return actions

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
        rewards = torch.Tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))
        rewards_mean = torch.mean(rewards, dim=1, keepdim=True)
        rewards_std = torch.std(rewards, dim=1, keepdim=True, unbiased=True)
        rewards = (rewards - rewards_mean) / (rewards_std + 1e-10)
        actions = torch.Tensor(actions, dtype=torch.int8, device=self.device)

        old_probs = torch.Tensor(old_probs, dtype=torch.float32, device=self.device)
        # convert states to policy (or probability)
        new_probs = pong_utils.states_to_prob(self.policy, states)
        new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)
        ratio = new_probs / (old_probs + 1.e-10)
        clip_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        ratio = torch.min(ratio, clip_ratio)
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                    (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))
        return torch.mean(ratio * rewards + beta * entropy)
