import torch
from network import A2CNet
from components import get_ac_cfg_defaults, Task, Storage
import numpy as np
from components import to_np, tensor
import adabound

hyper_parameter = get_ac_cfg_defaults().HYPER_PARAMETER.clone()
train_parameter = get_ac_cfg_defaults().TRAIN_PARAMETER.clone()


class A2CAgent(torch.nn.Module):
    def __init__(self, env_path):
        super(A2CAgent, self).__init__()
        self.policy = A2CNet()
        self.task = Task('Reacher', 20, env_path)
        self.states = self.task.reset().vector_observations
        self.rollout_length = hyper_parameter.ROLLOUT_LENGTH
        self.online_rewards = np.zeros(hyper_parameter.AGENTS_NUM)
        self.episode_rewards = []
        self.total_steps = 0
        self.optimizer = torch.optim.RMSprop(self.policy.total_params, lr=0.0001)
        # self.optimizer = torch.optim.Adam(self.policy.total_params, lr=1e-4, eps=1e-5)
        # self.optimizer = adabound.AdaBound(self.policy.total_params, lr=1e-4, final_lr=1.0)

    def step(self):
        storage = Storage(self.rollout_length)
        states = self.states
        for _ in range(self.rollout_length):
            prediction = self.policy(states)
            next_states, rewards, terminals = self.task.step(to_np(prediction['a']))
            self.online_rewards += rewards
            for i, terminal in enumerate(terminals):
                if terminal:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states
        self.states = states
        prediction = self.policy(states)
        storage.add(prediction)
        storage.placeholder()

        returns = prediction['v'].detach()
        advantages = tensor(np.zeros((hyper_parameter.AGENTS_NUM, 1)))
        for i in reversed(range(self.rollout_length)):
            returns = storage.r[i] + hyper_parameter.GAMMA * storage.m[i] * returns
            if not hyper_parameter.USE_GAE:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + hyper_parameter.GAMMA * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * hyper_parameter.GAE_TAU * hyper_parameter.GAMMA * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - train_parameter.entropy_weight * entropy_loss +
         train_parameter.value_loss_weight * value_loss).backward()
        # for param in self.policy.total_params:
        #     print(param.name, end='   ')
        # print()
        torch.nn.utils.clip_grad_norm_(self.policy.total_params, train_parameter.Gradient_Clip)
        self.optimizer.step()

        steps = self.rollout_length * hyper_parameter.AGENTS_NUM
        self.total_steps += steps

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def close(self):
        self.task.close()



