from components import Task
import numpy as np

env_path = './Reacher_Env/multiple_agents/Reacher_Linux/Reacher.x86_64'
if __name__ == '__main__':
    task = Task('Reacher', 20, env_path)
    task.reset()
    while True:
        action = np.random.rand(task.num_agents, task.action_dim)
        next_state, reward, done = task.step(action)