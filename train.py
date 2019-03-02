from agent import A2CAgent
from components import get_ac_cfg_defaults
import time
import numpy as np

hyper_parameter = get_ac_cfg_defaults().HYPER_PARAMETER.clone()
env_path = './Reacher_Env/multiple_agents/Reacher_Linux/Reacher.x86_64'

if __name__ == '__main__':
    agent = A2CAgent(env_path)
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if hyper_parameter.SAVE_INTERVAL and not agent.total_steps % hyper_parameter.SAVE_INTERVAL and len(agent.episode_rewards):
            agent.save('saved_models/model-%s-%s.pth' % (agent_name, str(agent.total_steps)))
        if hyper_parameter.LOG_INTERVAL and not agent.total_steps % hyper_parameter.LOG_INTERVAL and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            print('total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                agent.total_steps, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards),
                hyper_parameter.LOG_INTERVAL / (time.time() - t0)))
            t0 = time.time()
        if hyper_parameter.MAX_STEPS and agent.total_steps >= hyper_parameter.MAX_STEPS:
            agent.close()
            break
        agent.step()