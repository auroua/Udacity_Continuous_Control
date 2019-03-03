### This repo is the Udacity Deep Reinforcement Learning course Continue Control Project
> The README should be designed for a general audience that may not be familiar with the Nanodegree program; you should describe the environment that you solved, along with how to install the requirements before running the code in your repository.

#### Steps
1. install python
2. ```pip install pytorch``` reference [pytorch website](https://pytorch.org/)
3. ```pip install unityagents```
4. download **reacher env** from [Reacher_Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
5. Run ```components/env_tst.py``` to test the environment work all right.
6. Run ```train.py``` to train the agent. This module is for the vector state space.
7. Run ```test.py``` to test the agent interact with Env.
8. The default config file is  ```components/config_ac2.py```, ```components/config_ppo.py```, and ```components/config_ppo_ac.py```. You can modify the default parameter value to retrain the agent.


#### Code Environments
* XUbuntu 18.04
* CUDA 10.0
* cudnn 7.4.1
* Python 3.6
* Pytorch 1.0
* yacs v0.1.5


#### Reacher Env
* action space: 4 continuous action.
* state space: 33 states
* [version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30. 

#### TO-DO-LIST
* ~~PPO without using actor critic.~~
* ~~PPO with actor critic.~~
* ~~AC2~~

#### Project Architecture
* Package agent contains the PPO, AC2, PPO_AC agent.
* Package components contains the config files for different agent, envs and util functions.
* Package network contains the agent policy network.


#### References
1. [AdaBound](https://github.com/Luolc/AdaBound)
2. [DeepRL](https://github.com/ShangtongZhang/DeepRL)